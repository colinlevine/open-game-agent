"""
PyAutoGUI MCP Client

This client connects to the PyAutoGUI MCP server and provides an interactive
interface for GUI automation using Claude.
"""

import asyncio
import sys
import os
import argparse
from typing import Optional
from contextlib import AsyncExitStack


import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # SYSTEM_DPI_AWARE
except:
    pass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AzureOpenAI
from dotenv import load_dotenv

# Import game agent prompts
from constants import format_game_agent_prompt, GAME_AGENT_POLICY, DEFAULT_GAME_INSTRUCTIONS, select_game_instructions
from thought_overlay import ThoughtOverlay
from knowledge_integration import KnowledgeEnhancedGameAgent

# Load environment variables from .env
load_dotenv()


class PyAutoGUIClient:
    def __init__(self, requests_per_minute: int = 6, overlay: ThoughtOverlay = None):
        """Initialize the MCP client for PyAutoGUI automation."""
        self.azure_openai = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        
        # Control frequency of game agent decisions (requests per minute)
        self.requests_per_minute = requests_per_minute
        self.loop_delay = 60.0 / self.requests_per_minute  # Calculate delay in seconds between iterations
        
        # Game state tracking for prompts
        self.user_instructions = DEFAULT_GAME_INSTRUCTIONS
        self.game_state = "Unknown - analyze from current visual state"
        self.past_decisions = "No previous decisions recorded"
        self.learned_patterns = "No patterns learned yet"
        # Optional overlay for model thoughts
        self.overlay = overlay

        # Initialize knowledge graph integration
        self.knowledge_agent = None
        self.knowledge_enabled = os.getenv("AI_SEARCH_ENABLED", "false").lower() == "true"

    async def connect_to_server(self, server_script_path: str):
        """Connect to the PyAutoGUI MCP server.
        
        Args:
            server_script_path: Path to the server.py script
        """
        if not server_script_path.endswith('.py'):
            raise ValueError("Server script must be a .py file")
        
        if not os.path.exists(server_script_path):
            raise ValueError(f"Server script not found: {server_script_path}")

        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(f"\nConnected to PyAutoGUI MCP server with tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Initialize AI Search if enabled
        if self.knowledge_enabled:
            try:
                print("🧠 Initializing AI Search...")
                from knowledge_integration import KnowledgeEnhancedGameAgent
                self.knowledge_agent = KnowledgeEnhancedGameAgent(self, learning_enabled=True)
                await self.knowledge_agent.initialize()
                print("✅ Knowledge graph initialized successfully")
            except Exception as e:
                print(f"⚠️ Knowledge graph initialization failed: {e}")
                self.knowledge_enabled = False

    async def process_query(self, query: str, is_initial_instruction: bool = False) -> str:
        """Process a query using Azure OpenAI GPT-4o-mini and available PyAutoGUI tools.
        
        Args:
            query: User query for GUI automation
            is_initial_instruction: Whether this is the initial user instruction
            
        Returns:
            Response from GPT-4o-mini after potentially using tools
        """
        # Handle initial user instructions
        if is_initial_instruction:
            self.user_instructions = query
        
        # Automatically capture visual state locally
        # Capture screenshot locally and encode as base64
        # Hide overlay to avoid capturing it
        if self.overlay:
            self.overlay.hide()
        import pyautogui, io, base64
        base64_image = None
        try:
            screenshot = pyautogui.screenshot()
            buf = io.BytesIO()
            screenshot.save(buf, format='PNG')
            base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            print("📸 Visual state captured.")
        except Exception as e:
            print(f"❌ Failed to capture visual state: {e}")
        # Restore overlay after screenshot
        if self.overlay:
            self.overlay.show()

        # Get available tools from the server
        tools_response = await self.session.list_tools()
        available_tools = []
        
        for tool in tools_response.tools:
            # Exclude any screenshot tool if present
            if tool.name.lower() == "screenshot":
                continue
            available_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
        
        # Format tools for prompt context
        tools_context = "\n".join([
            f"- {tool['function']['name']}: {tool['function']['description']}"
            for tool in available_tools
        ])
        
        # Get enhanced system prompt if knowledge graph is available
        if self.knowledge_enabled and self.knowledge_agent:
            base_system_prompt = format_game_agent_prompt(
                user_instructions=self.user_instructions,
                game_state=self.game_state,
                past_decisions=self.past_decisions,
                learned_patterns=self.learned_patterns,
                available_tools=tools_context
            )
            system_prompt = await self.knowledge_agent.get_knowledge_enhanced_prompt(
                base_prompt=base_system_prompt,
                game_state=self.game_state
            )
        else:
            system_prompt = format_game_agent_prompt(
                user_instructions=self.user_instructions,
                game_state=self.game_state,
                past_decisions=self.past_decisions,
                learned_patterns=self.learned_patterns,
                available_tools=tools_context
            )

        # Build messages with system prompt and user query that includes current visual state
        user_content = [
            {
                "type": "text",
                "text": f"Please analyze the current visual state and {query}"
            }
        ]

        if base64_image:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "auto"
                }
            })
        else:
            # Fallback if screenshot failed, provide text-only message
            user_content[0]['text'] = f"Current visual state: CAPTURE FAILED. Please proceed based on text context.\n\nAnalyze the current situation and {query}"

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        # Initial Azure OpenAI API call
        response = self.azure_openai.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            max_completion_tokens=1500
        )

        # Process response and handle tool calls
        final_text = []
        assistant_message = response.choices[0].message

        if assistant_message.content:
            final_text.append(assistant_message.content)

        if assistant_message.tool_calls:
            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in assistant_message.tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                tool_call_id = tool_call.id

                print(f"\n🔧 Executing tool: {tool_name}")
                print(f"📝 Arguments: {tool_args_str}")

                try:
                    # Parse the JSON arguments
                    import json
                    tool_args = json.loads(tool_args_str)

                    # Call the tool via MCP
                    tool_response = await self.session.call_tool(tool_name, tool_args)
                    tool_result = ""
                    
                    for result_content in tool_response.content:
                        if hasattr(result_content, 'text'):
                            tool_result += result_content.text

                    print(f"✅ Tool result: {tool_result}")

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result
                    })

                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(f"❌ {error_msg}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": error_msg
                    })

            # Get GPT's final response after tool execution
            final_response = self.azure_openai.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_completion_tokens=1500
            )

            if final_response.choices[0].message.content:
                final_text.append(final_response.choices[0].message.content)

        final_result = "\n".join(final_text)
        # Update overlay with AI thought
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.update(final_result)
        
        # Record action in knowledge graph if enabled
        if self.knowledge_enabled and self.knowledge_agent and not is_initial_instruction:
            try:
                await self.knowledge_agent.record_game_action(
                    game_state=self.game_state,
                    action_taken=query,
                    immediate_outcome=final_result[:200] + "..." if len(final_result) > 200 else final_result
                )
            except Exception as e:
                print(f"⚠️ Failed to record action in knowledge graph: {e}")
        
        return final_result

    async def chat_loop(self):        
        """Run an automated game agent loop for GUI automation."""
        print("\n🎮 AI Game Agent (Azure OpenAI GPT-4.1-mini)")
        print("=" * 60)
        print("🎯 Automated Game Agent:")
        print("1. Agent captures visual state and makes decisions automatically")
        print("2. Agent analyzes current state and follows predefined instructions")
        print("3. Game instructions are set in constants.py (DEFAULT_GAME_INSTRUCTIONS)")
        print("4. Agent runs continuously at configurable intervals")
        print(f"\n⏱️  Loop frequency: {self.requests_per_minute} requests per minute ({self.loop_delay:.1f}s between iterations)")
        print("\nPress Ctrl+C to stop.")
        print("=" * 60)        # Process default instructions on startup
        print(f"\n📋 Using default instructions: {self.user_instructions}")
        response = await self.process_query("Analyze the current game state and begin following the user instructions.", is_initial_instruction=True)
        print(f"\n🤖 Agent: {response}")

        # Update game state based on initial analysis to provide memory
        self.past_decisions = f"Initial analysis and action: {response}"
        self.game_state = "Game is in progress. Analyzing results of the first action."

        iteration_count = 0
        while True:
            try:
                # Wait before next iteration based on requests_per_minute setting
                if self.loop_delay > 0 and iteration_count > 0:
                    print(f"\n⏳ Waiting {self.loop_delay:.1f} seconds before next iteration...")
                    await asyncio.sleep(self.loop_delay)
                
                iteration_count += 1
                print(f"\n🔄 Iteration #{iteration_count}")
                
                # Automatic game agent decision making
                print("💭 Agent analyzing game state...")
                agent_query = "Analyze the current game state and make the next strategic decision based on your instructions and past decisions."
                
                print("🤔 Processing game state...")
                response = await self.process_query(agent_query)
                print(f"\n🤖 Agent Decision: {response}")

                # Update game state for the next iteration to provide memory
                self.past_decisions += f"\n\n--- End of Previous Iteration ---\n\nDecision at Iteration {iteration_count}: {response}"
                # To prevent the context from growing too large, we could summarize, but for now, appending is fine.
                if response and response.strip():
                    self.game_state = f"After iteration {iteration_count}, the agent decided: {response.splitlines()[0]}" # Use first line as a summary
                else:
                    self.game_state = f"After iteration {iteration_count}, no specific decision text was returned."

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                # Still wait on error to maintain timing
                if self.loop_delay > 0:
                    await asyncio.sleep(self.loop_delay)
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.stop()
        
        # Finalize knowledge graph session if enabled
        if self.knowledge_enabled and self.knowledge_agent:
            try:
                await self.knowledge_agent.finalize_game_session(
                    final_outcome="Game session ended",
                    success=True,  # You might want to track actual success
                    strategy_type="general"
                )
                print("💾 Knowledge graph session finalized")
            except Exception as e:
                print(f"⚠️ Failed to finalize knowledge graph session: {e}")
        
        await self.exit_stack.aclose()

async def main():
    """Main entry point for the AI Game Agent client."""
    parser = argparse.ArgumentParser(description="AI Game Agent - Automated game playing using PyAutoGUI")
    parser.add_argument("server_path", help="Path to the server.py script")
    parser.add_argument("--rpm", type=int, default=6, 
                       help="Requests per minute for game agent decisions (default: 6)")
    parser.add_argument("--game", type=str, choices=["wordle", "2048"], 
                       help="Specify game directly without interactive selection")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.server_path):
        print(f"Error: Server script not found: {args.server_path}")
        print("\nUsage: python client.py <path_to_server.py> [--rpm=REQUESTS_PER_MINUTE] [--game=GAME_NAME]")
        print("Example: python client.py server.py --rpm=10 --game=wordle")
        print("\nThis starts an AI Game Agent that:")
        print("- Captures visual state at regular intervals")
        print("- Analyzes game state using AI")
        print("- Makes strategic decisions based on selected game instructions")
        print("- Supports Wordle and 2048 games")
        sys.exit(1)

    # Start thought overlay
    overlay = ThoughtOverlay()
    overlay.start()
    client = PyAutoGUIClient(requests_per_minute=args.rpm, overlay=overlay)

    # Set game instructions based on selection
    if args.game:
        from constants import AVAILABLE_GAMES
        if args.game in AVAILABLE_GAMES:
            selected_instructions = AVAILABLE_GAMES[args.game]['instructions']
            print(f"\n✅ Using {AVAILABLE_GAMES[args.game]['name']} instructions from command line")
        else:
            print(f"❌ Invalid game specified: {args.game}")
            sys.exit(1)
    else:
        # Interactive game selection
        selected_instructions = select_game_instructions()
    
    # Set the selected instructions
    client.user_instructions = selected_instructions

    try:
        await client.connect_to_server(args.server_path)
        await client.chat_loop()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
