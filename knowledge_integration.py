"""
Knowledge Graph Integration for Game Agent

This module integrates the Azure AI Search Knowledge Graph with the existing
PyAutoGUI game agent to enhance decision-making through learned experiences.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any
from azure_ai_search import AzureAISearchKnowledgeGraph, GameKnowledgeNode

logger = logging.getLogger(__name__)


class KnowledgeEnhancedGameAgent:
    """
    Enhanced game agent that uses Azure AI Search Knowledge Graph
    to make better decisions based on past experiences.
    """
    
    def __init__(self, 
                 pyautogui_client,
                 knowledge_graph: AzureAISearchKnowledgeGraph = None,
                 learning_enabled: bool = True):
        """
        Initialize the knowledge-enhanced game agent.
        
        Args:
            pyautogui_client: The existing PyAutoGUIClient instance
            knowledge_graph: Optional pre-configured knowledge graph instance
            learning_enabled: Whether to learn from experiences
        """
        self.client = pyautogui_client
        self.learning_enabled = learning_enabled
        
        # Initialize knowledge graph if not provided
        if knowledge_graph is None:
            # Get index name from environment variable, with fallback
            index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "game-agent-knowledge")
            self.knowledge_graph = AzureAISearchKnowledgeGraph(
                index_name=index_name
            )
        else:
            self.knowledge_graph = knowledge_graph
        
        # Track current session data
        self.current_session = {
            "states": [],
            "actions": [],
            "outcomes": [],
            "session_id": None
        }
        
    async def initialize(self):
        """Initialize the knowledge graph and any required resources."""
        try:
            await self.knowledge_graph.initialize_index()
            logger.info("Knowledge-enhanced game agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
            # Disable learning if initialization fails
            self.learning_enabled = False
    
    async def enhance_game_state_analysis(self, 
                                        game_state: str,
                                        visual_data: str = None) -> Dict[str, Any]:
        """
        Enhance game state analysis with knowledge graph insights.
        
        Args:
            game_state: Current game state description
            visual_data: Optional visual data from screenshot
            
        Returns:
            Enhanced analysis with recommendations and insights
        """
        enhanced_analysis = {
            "current_state": game_state,
            "recommendations": [],
            "similar_experiences": [],
            "pattern_insights": {},
            "confidence_score": 0.0
        }
        
        if not self.learning_enabled:
            return enhanced_analysis
        
        try:
            # Get strategic recommendations based on current state
            recommendations = await self.knowledge_graph.get_strategic_recommendations(
                current_state=game_state,
                strategy_type=None  # Get all strategy types
            )
            
            enhanced_analysis["recommendations"] = recommendations
            
            # Find similar past experiences
            similar_states = await self.knowledge_graph.search_similar_knowledge(
                query=game_state,
                node_types=["game_state", "outcome"],
                top_k=5
            )
            
            enhanced_analysis["similar_experiences"] = similar_states
            
            # Get pattern insights
            patterns = await self.knowledge_graph.analyze_patterns()
            enhanced_analysis["pattern_insights"] = patterns
            
            # Calculate confidence score based on available data
            if recommendations:
                avg_success_rate = sum(r["success_rate"] for r in recommendations) / len(recommendations)
                avg_confidence = sum(r["confidence"] for r in recommendations) / len(recommendations)
                enhanced_analysis["confidence_score"] = (avg_success_rate + avg_confidence) / 2
            
            logger.info(f"Enhanced analysis generated with {len(recommendations)} recommendations")
            
        except Exception as e:
            logger.error(f"Error enhancing game state analysis: {e}")
        
        return enhanced_analysis
    
    async def get_knowledge_enhanced_prompt(self, 
                                          base_prompt: str,
                                          game_state: str) -> str:
        """
        Enhance the game agent prompt with knowledge graph insights.
        
        Args:
            base_prompt: The original system prompt
            game_state: Current game state
            
        Returns:
            Enhanced prompt with knowledge graph insights
        """
        if not self.learning_enabled:
            return base_prompt
        
        try:
            # Get enhanced analysis
            analysis = await self.enhance_game_state_analysis(game_state)
            
            # Build knowledge enhancement section
            knowledge_section = "\n\n## 🧠 KNOWLEDGE GRAPH INSIGHTS\n\n"
            
            # Add recommendations
            if analysis["recommendations"]:
                knowledge_section += "### 📊 Strategic Recommendations (Based on Past Experiences):\n"
                for i, rec in enumerate(analysis["recommendations"][:3], 1):
                    knowledge_section += f"{i}. **Action:** {rec['action']}\n"
                    knowledge_section += f"   **Success Rate:** {rec['success_rate']:.1%}\n"
                    knowledge_section += f"   **Confidence:** {rec['confidence']:.2f}\n"
                    knowledge_section += f"   **Strategy Type:** {rec['strategy_type']}\n\n"
            
            # Add pattern insights
            if analysis["pattern_insights"].get("most_effective_patterns"):
                knowledge_section += "### 🔍 Most Effective Patterns:\n"
                for action, count in analysis["pattern_insights"]["most_effective_patterns"][:3]:
                    knowledge_section += f"- **{action}** (used successfully {count} times)\n"
                knowledge_section += "\n"
            
            # Add similar experiences
            if analysis["similar_experiences"]:
                knowledge_section += "### 📝 Similar Past Experiences:\n"
                for exp in analysis["similar_experiences"][:2]:
                    knowledge_section += f"- **{exp['type']}:** {exp['content'][:100]}...\n"
                    knowledge_section += f"  *Relevance Score: {exp['score']:.2f}*\n\n"
            
            # Add confidence assessment
            confidence_score = analysis["confidence_score"]
            if confidence_score > 0:
                confidence_level = "High" if confidence_score > 0.7 else "Medium" if confidence_score > 0.4 else "Low"
                knowledge_section += f"### 🎯 Knowledge Confidence: {confidence_level} ({confidence_score:.1%})\n\n"
            
            knowledge_section += "**💡 Use these insights to inform your decision-making process.**\n"
            knowledge_section += "**⚠️ Always prioritize current visual analysis over historical patterns.**\n\n"
            
            # Combine with base prompt
            enhanced_prompt = base_prompt + knowledge_section
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt with knowledge: {e}")
            return base_prompt
    
    async def record_game_action(self, 
                               game_state: str,
                               action_taken: str,
                               immediate_outcome: str = None):
        """
        Record a game action for future learning.
        
        Args:
            game_state: The game state when action was taken
            action_taken: Description of the action taken
            immediate_outcome: Immediate result of the action
        """
        if not self.learning_enabled:
            return
        
        try:
            # Store in current session
            self.current_session["states"].append(game_state)
            self.current_session["actions"].append(action_taken)
            if immediate_outcome:
                self.current_session["outcomes"].append(immediate_outcome)
            
            logger.info(f"Recorded action: {action_taken[:50]}...")
            
        except Exception as e:
            logger.error(f"Error recording game action: {e}")
    
    async def finalize_game_session(self, 
                                  final_outcome: str,
                                  success: bool,
                                  strategy_type: str = "general"):
        """
        Finalize a game session and add complete experience to knowledge graph.
        
        Args:
            final_outcome: Final result of the game session
            success: Whether the session was successful
            strategy_type: Type of strategy used
        """
        if not self.learning_enabled or not self.current_session["states"]:
            return
        
        try:
            # Combine session data into a comprehensive experience
            session_summary = {
                "states": self.current_session["states"],
                "actions": self.current_session["actions"],
                "outcomes": self.current_session["outcomes"],
                "final_outcome": final_outcome,
                "success": success,
                "strategy_type": strategy_type
            }
            
            # Add the complete experience to knowledge graph
            for i, (state, action) in enumerate(zip(session_summary["states"], session_summary["actions"])):
                # Determine outcome for this step
                step_outcome = session_summary["outcomes"][i] if i < len(session_summary["outcomes"]) else "Step completed"
                
                # For the last step, use final outcome
                if i == len(session_summary["states"]) - 1:
                    step_outcome = final_outcome
                
                # Determine if this step was successful
                step_success = success if i == len(session_summary["states"]) - 1 else "intermediate"
                
                await self.knowledge_graph.add_game_experience(
                    game_state=state,
                    action_taken=action,
                    outcome=step_outcome,
                    success=step_success if isinstance(step_success, bool) else True,
                    strategy_type=strategy_type
                )
            
            logger.info(f"Finalized game session with {len(session_summary['states'])} steps")
            
            # Reset session
            self.current_session = {
                "states": [],
                "actions": [],
                "outcomes": [],
                "session_id": None
            }
            
        except Exception as e:
            logger.error(f"Error finalizing game session: {e}")
    
    async def get_adaptive_strategy_suggestions(self, 
                                              current_performance: Dict[str, Any]) -> List[str]:
        """
        Get adaptive strategy suggestions based on current performance and knowledge graph.
        
        Args:
            current_performance: Dictionary with performance metrics
            
        Returns:
            List of strategy suggestions
        """
        suggestions = []
        
        if not self.learning_enabled:
            return ["Knowledge graph learning is disabled"]
        
        try:
            # Analyze current performance
            success_rate = current_performance.get("success_rate", 0)
            avg_attempts = current_performance.get("avg_attempts", 0)
            
            # Get pattern analysis
            patterns = await self.knowledge_graph.analyze_patterns()
            
            # Generate suggestions based on performance and patterns
            if success_rate < 0.5:
                suggestions.append("🔄 Consider switching to more proven strategies")
                
                # Find most successful strategies
                if patterns.get("success_rates_by_strategy"):
                    best_strategies = sorted(
                        patterns["success_rates_by_strategy"].items(),
                        key=lambda x: x[1].get("rate", 0),
                        reverse=True
                    )
                    
                    if best_strategies:
                        best_strategy = best_strategies[0][0]
                        best_rate = best_strategies[0][1].get("rate", 0)
                        suggestions.append(f"🎯 Try '{best_strategy}' strategy (success rate: {best_rate:.1%})")
            
            if avg_attempts > 4:
                suggestions.append("⚡ Focus on more efficient opening moves")
                
                # Suggest most effective patterns
                if patterns.get("most_effective_patterns"):
                    top_pattern = patterns["most_effective_patterns"][0][0]
                    suggestions.append(f"💡 Consider using: {top_pattern}")
            
            # Add general insights
            total_experiences = patterns.get("total_successful_experiences", 0)
            if total_experiences > 10:
                suggestions.append(f"📈 Learning from {total_experiences} successful experiences")
            else:
                suggestions.append("🌱 Building knowledge base - continue playing to improve recommendations")
            
        except Exception as e:
            logger.error(f"Error getting adaptive strategy suggestions: {e}")
            suggestions.append("❌ Error analyzing strategies")
        
        return suggestions
    
    async def export_knowledge_report(self, output_path: str = "knowledge_report.json"):
        """Export a comprehensive knowledge report."""
        if not self.learning_enabled:
            return
        
        try:
            # Get comprehensive analysis
            patterns = await self.knowledge_graph.analyze_patterns()
            
            # Search for all knowledge nodes
            all_strategies = await self.knowledge_graph.search_similar_knowledge(
                query="strategy",
                node_types=["action", "strategy"],
                top_k=50
            )
            
            report = {
                "timestamp": asyncio.get_event_loop().time(),
                "patterns": patterns,
                "top_strategies": all_strategies[:10],
                "knowledge_graph_stats": {
                    "total_nodes": len(self.knowledge_graph.graph.nodes()),
                    "total_edges": len(self.knowledge_graph.graph.edges()),
                    "node_types": {}
                }
            }
            
            # Count node types
            for node_id, node_data in self.knowledge_graph.graph.nodes(data=True):
                node_type = node_data.get("type", "unknown")
                report["knowledge_graph_stats"]["node_types"][node_type] = \
                    report["knowledge_graph_stats"]["node_types"].get(node_type, 0) + 1
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Knowledge report exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting knowledge report: {e}")


# Integration helper functions
async def integrate_with_existing_client(pyautogui_client):
    """
    Helper function to integrate knowledge graph with existing PyAutoGUIClient.
    
    This shows how to modify the existing client to use knowledge graph insights.
    """
    # Create knowledge-enhanced agent
    kg_agent = KnowledgeEnhancedGameAgent(pyautogui_client)
    await kg_agent.initialize()
    
    # Store reference in the original client
    pyautogui_client.knowledge_agent = kg_agent
    
    # Enhance the process_query method
    original_process_query = pyautogui_client.process_query
    
    async def enhanced_process_query(query: str, is_initial_instruction: bool = False) -> str:
        """Enhanced process_query that uses knowledge graph insights."""
        # Get current game state for knowledge lookup
        current_state = pyautogui_client.game_state
        
        # Record the action if this is not the initial instruction
        if not is_initial_instruction and hasattr(pyautogui_client, '_last_query'):
            await kg_agent.record_game_action(
                game_state=current_state,
                action_taken=pyautogui_client._last_query,
                immediate_outcome="Action executed"
            )
        
        # Store current query for next iteration
        pyautogui_client._last_query = query
        
        # Get enhanced prompt if knowledge graph is available
        enhanced_prompt = None
        if hasattr(pyautogui_client, 'knowledge_agent'):
            # Get the system prompt (you might need to modify constants.py to export this)
            from constants import format_game_agent_prompt
            base_system_prompt = format_game_agent_prompt(
                user_instructions=pyautogui_client.user_instructions,
                game_state=pyautogui_client.game_state,
                past_decisions=pyautogui_client.past_decisions,
                learned_patterns=pyautogui_client.learned_patterns,
                available_tools=""  # Will be filled in original method
            )
            
            enhanced_prompt = await kg_agent.get_knowledge_enhanced_prompt(
                base_prompt=base_system_prompt,
                game_state=current_state
            )
        
        # Call original method (might need modification to accept enhanced prompt)
        result = await original_process_query(query, is_initial_instruction)
        
        return result
    
    # Replace the method
    pyautogui_client.process_query = enhanced_process_query
    
    return kg_agent


# Example of how to modify the chat_loop to include knowledge graph finalization
async def enhanced_chat_loop_example(pyautogui_client):
    """
    Example of how to modify the chat loop to include knowledge graph learning.
    This is a template - you would integrate this into the existing chat_loop method.
    """
    # Assume we have integrated the knowledge agent
    kg_agent = pyautogui_client.knowledge_agent
    
    try:
        # ... existing chat loop code ...
        
        # At the end of a game session, finalize the knowledge
        await kg_agent.finalize_game_session(
            final_outcome="Game completed successfully",  # or failure message
            success=True,  # or False based on actual result
            strategy_type="wordle_strategy"  # or determine dynamically
        )
        
        # Get adaptive suggestions for next session
        current_performance = {
            "success_rate": 0.75,  # Calculate from recent games
            "avg_attempts": 4.2   # Calculate from recent games
        }
        
        suggestions = await kg_agent.get_adaptive_strategy_suggestions(current_performance)
        
        print("\n🎓 Adaptive Strategy Suggestions:")
        for suggestion in suggestions:
            print(f"  {suggestion}")
        
    except Exception as e:
        logger.error(f"Error in enhanced chat loop: {e}")


if __name__ == "__main__":
    # Test the knowledge-enhanced agent
    async def test():
        from azure_ai_search import AzureAISearchKnowledgeGraph
        
        # Create a test knowledge graph
        kg = AzureAISearchKnowledgeGraph(index_name="test-enhanced-agent")
        
        # Create a mock client (you would use your actual PyAutoGUIClient)
        class MockClient:
            def __init__(self):
                self.game_state = "Test game state"
                self.user_instructions = "Test instructions"
                self.past_decisions = "Test decisions"
                self.learned_patterns = "Test patterns"
        
        mock_client = MockClient()
        
        # Create enhanced agent
        enhanced_agent = KnowledgeEnhancedGameAgent(mock_client, kg)
        await enhanced_agent.initialize()
        
        # Test recording actions
        await enhanced_agent.record_game_action(
            game_state="Game started",
            action_taken="Made opening move",
            immediate_outcome="Move accepted"
        )
        
        # Test finalizing session
        await enhanced_agent.finalize_game_session(
            final_outcome="Game won",
            success=True,
            strategy_type="test_strategy"
        )
        
        print("Knowledge-enhanced agent test completed!")
    
    asyncio.run(test())
