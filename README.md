# Open Game Agent

A Model Context Protocol (MCP) implementation for GUI automation using PyAutoGUI. This project provides a **generalized game agent framework** that helps developers create vector-enhanced memory systems to enable their AI agents to make more intelligent decisions based on past experiences.

## Overview

This framework demonstrates how to build AI agents that learn from their actions and improve over time. The current demo supports **2048** and **Wordle** games out of the box, but developers can extend it to work with any game by providing appropriate prompts and game-specific logic.

**Note**: Wordle support is work-in-progress and may not always be consistent. The 2048 implementation is more stable.

**Key Benefits for Developers:**
- Learn how to integrate Azure AI Search for basic vector storage
- See examples of how AI agents can store and retrieve past decisions
- Understand how to enhance AI prompts with historical context
- Get a basic foundation that can be extended for more sophisticated learning systems

## Features

### Basic Vector Search System (Experimental)

The game agent includes a simple vector search system powered by Azure AI Search for demonstration purposes:

- **Basic vector storage**: Stores game actions and outcomes as embeddings
- **Simple similarity search**: Finds potentially relevant past situations
- **Basic pattern storage**: Records successful and failed approaches
- **Cross-session persistence**: Saves data between game sessions
- **Prompt enhancement**: Adds relevant historical context to AI prompts

**Note**: This is a basic demonstration system that requires significant customization for real-world use. It's designed as a starting point for developers to build more sophisticated learning systems.

### Legacy Memory System (Optional)

Also includes a basic memory system for simple use cases:

- **Cross-session persistence**: Remembers strategies across game sessions
- **Pattern learning**: Identifies successful and failed approaches
- **Contextual recall**: Retrieves relevant past decisions for current situations
- **Human-readable storage**: All memory stored as markdown files you can view/edit

### Available Tools

1. **Click** - Click at screen coordinates using percentage values (0.0 to 1.0) ⚠️ *Work in Progress*
   - Maps percentage coordinates to actual screen dimensions
   - Supports different mouse buttons (left, right, middle)
   - Configurable click count and interval
   - *Note: Currently being refined for better accuracy and reliability*

2. **Keypress** - Send keyboard input to the active window
   - Single characters and special keys
   - Key combinations (Ctrl+C, Alt+Tab, etc.)
   - Key sequences (space-separated keys)
   - Text input with configurable intervals
   - Smart key parsing for concatenated key names

3. **Get Screen Info** - Get screen dimensions and current mouse position
   - Screen resolution
   - Current mouse coordinates (pixels and percentages)
   - Helpful for determining click coordinates

## Setup

### Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux with GUI support

### Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your Azure OpenAI and Azure AI Search settings
   ```

## Environment Configuration

This project requires several Azure services. Create a `.env` file with the following configuration:

### Required Azure OpenAI Settings
```env
# Azure OpenAI Configuration (Required)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini

# Azure OpenAI Embeddings for AI Search
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# API request rate limiting (Optional - default: 6)
REQUESTS_PER_MINUTE=6
```

### Required Azure AI Search Settings
```env
# Azure AI Search Configuration
AZURE_SEARCH_SERVICE_NAME=your-search-service-name
AZURE_SEARCH_API_KEY=your_search_api_key_here

# Alternative: Use Managed Identity instead of API keys (advanced)
AZURE_USE_MANAGED_IDENTITY=false
```

### Optional Settings
```env
# Vector Search Features (Optional - set to "true" to enable)
AI_SEARCH_ENABLED=true

# Azure AI Search Index Name (Optional - customize for different environments)
# Examples: "my-game-agent", "wordle-agent-dev", "game-ai-prod"
AZURE_SEARCH_INDEX_NAME=game-agent-knowledge
```

### How to Get These Values

1. **Azure OpenAI**:
   - Create an Azure OpenAI resource in the Azure portal
   - Deploy a gpt-4.1-mini model (or whichever OpenAI model you choose) and a text-embedding-ada-002 model
   - Get the API key and endpoint from the resource overview
   - Use the deployment names you created

2. **Azure AI Search**:
   - Create an Azure AI Search service in the Azure portal
   - Get the service name and admin key from the service overview
   - Ensure the service tier supports vector search (Basic or higher)

3. **Enable Features**:
   - Set `AI_SEARCH_ENABLED=true` to use the basic AI learning system
   - You can disable it by setting it to `false` if you don't want to use Azure AI Search

4. **Customize Index Name** (Optional):
   - Set `AZURE_SEARCH_INDEX_NAME` to a unique name for your project
   - Use different names for development, staging, and production environments
   - Examples: `my-wordle-agent`, `game-ai-dev`, `prod-game-knowledge`
   - If not set, defaults to `game-agent-knowledge`

## Usage

### Running the AI Game Agent

The main way to use this project is with the AI game agent that automatically plays supported games:

```bash
python client.py server.py
```

When you run the client, you'll be prompted to choose which game you want the AI to play:

```
🎮 Welcome to the Game Agent!
==================================================
Available games:
  wordle: Wordle - 5-letter word guessing game with color-coded feedback ⚠️ *Work in Progress*
      🌐 Play at: https://www.nytimes.com/games/wordle/index.html or https://wordleunlimited.org/
      *Note: Wordle automation may be inconsistent and require manual intervention*
  2048: 2048 - Sliding puzzle game - combine tiles to reach 2048
      🌐 Play at: https://2048game.com/ (use this exact URL - other 2048 sites may not work)

Which game would you like to play?
Enter your choice (wordle/2048): wordle

⚠️  IMPORTANT: Make sure to click on the game window to select it as the primary window
   so that keyboard inputs will be sent to the correct game!
```

**Important Setup Steps:**

1. **Open the game in your browser** using the URLs provided above
⚠️ Other replica sites for 2048 and Wordle may require additional customization to work with this automation.
2. **Click on the game window** to make it the active/primary window
3. **Keep the game window visible** - don't minimize it or cover it completely
4. **Start the AI agent** and let it begin playing

**⚠️ Critical Window Focus Requirement:**
- The AI agent sends keyboard inputs to the **currently active window**
- Always ensure the game window is selected/clicked before starting the agent
- Keep the game window visible and in focus during gameplay
- If inputs seem to go to the wrong window, click on the game window to refocus

The AI agent sends keyboard inputs to the currently active window, so it's crucial that the game window remains the primary focus.

You can also specify the game directly via command line:

```bash
# Play Wordle (work-in-progress - may be inconsistent)
python client.py server.py --game=wordle

# Play 2048 (more stable)
python client.py server.py --game=2048

# Adjust AI decision frequency (default: 6 requests per minute)
python client.py server.py --game=wordle --rpm=10
```

### Adding Support for New Games

To add support for a new game:

1. **Create game instructions** in `constants.py`:
   ```python
   YOUR_GAME_INSTRUCTIONS = """
   # Your Game: Game Description & Controls
   [Add detailed instructions for your game]
   """
   ```

2. **Add to AVAILABLE_GAMES** dictionary:
   ```python
   AVAILABLE_GAMES = {
       # ... existing games ...
       "yourgame": {
           "name": "Your Game",
           "instructions": YOUR_GAME_INSTRUCTIONS,
           "description": "Brief description of your game"
       }
   }
   ```

3. **Update the command line choices** in `client.py` if needed

The AI agent will use the provided instructions to understand how to play your game!

### AI Search Features (Basic Implementation)

When `AI_SEARCH_ENABLED=true`, the client will:

1. **Initialize Azure AI Search index** for storing game actions
2. **Record game actions** with basic context and outcomes
3. **Add historical context** to AI prompts when similar situations are found
4. **Store basic patterns** from game sessions
5. **Provide simple recommendations** based on stored data

**Important**: This is a minimal implementation designed as a learning example. The knowledge graph:
- Has basic vector search without sophisticated ranking
- Provides simple pattern matching rather than advanced learning
- Requires significant customization for production use
- Is intended as a foundation for developers to build upon

This demonstrates:
- How to create embeddings for game states and actions
- Basic vector similarity search
- Simple success/failure tracking
- Cross-session data persistence

### Testing the AI search

Run the test suite to verify your Azure AI Search integration:

```bash
python test_ai_search.py
```

This will test basic functionality:
- Index creation and schema validation
- Document upload with embeddings
- Simple vector search capabilities
- Basic recommendation generation
- Pattern storage and retrieval
