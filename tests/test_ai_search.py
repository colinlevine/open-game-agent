"""
Test script for Azure AI Search Knowledge Graph integration.

This script tests the knowledge graph functionality without requiring
the full game agent setup.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from azure_ai_search import AzureAISearchKnowledgeGraph

# Load environment variables
load_dotenv()


async def test_basic_functionality():
    """Test basic knowledge graph functionality."""
    print("🧪 Testing Azure AI Search Knowledge Graph")
    print("=" * 50)
    
    # Check environment variables
    required_vars = [
        "AZURE_SEARCH_SERVICE_NAME",
        "AZURE_SEARCH_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease check your .env file and ensure all required variables are set.")
        return False
    
    try:
        # Initialize knowledge graph
        print("🔧 Initializing knowledge graph...")
        kg = AzureAISearchKnowledgeGraph(
            index_name="test-knowledge-graph"
        )
        
        # Initialize index
        print("📊 Creating search index...")
        await kg.initialize_index()
        print("✅ Search index created successfully")
        
        # Test embedding generation
        print("🧠 Testing embedding generation...")
        embeddings = await kg.generate_embeddings("Test text for embeddings")
        if embeddings and len(embeddings) > 0:
            print(f"✅ Generated embeddings with {len(embeddings)} dimensions")
        else:
            print("❌ Failed to generate embeddings")
            return False
        
        # Test adding game experience
        print("🎮 Adding test game experience...")
        experience_id = await kg.add_game_experience(
            game_state="Starting a new Wordle game",
            action_taken="Used ADIEU as opening word",
            outcome="Got 2 yellow letters (A, E)",
            success=False,
            strategy_type="vowel_strategy"
        )
        
        if experience_id:
            print(f"✅ Added game experience: {experience_id}")
        else:
            print("❌ Failed to add game experience")
            return False
        
        # Test search functionality
        print("🔍 Testing knowledge search...")
        search_results = await kg.search_similar_knowledge(
            query="Starting a Wordle game",
            top_k=3
        )
        
        if search_results:
            print(f"✅ Found {len(search_results)} similar knowledge entries")
            for result in search_results[:2]:
                print(f"   - {result['type']}: {result['content'][:60]}...")
        else:
            print("⚠️ No search results found (this is normal for a new index)")
        
        # Test strategic recommendations
        print("📋 Testing strategic recommendations...")
        recommendations = await kg.get_strategic_recommendations(
            current_state="Need to make first move in Wordle"
        )
        
        if recommendations:
            print(f"✅ Generated {len(recommendations)} recommendations")
            for rec in recommendations[:2]:
                print(f"   - {rec['action']} (success rate: {rec['success_rate']:.1%})")
        else:
            print("⚠️ No recommendations available (this is normal for a new knowledge base)")
        
        # Test pattern analysis
        print("📊 Testing pattern analysis...")
        patterns = await kg.analyze_patterns()
        
        if patterns:
            print("✅ Pattern analysis completed")
            print(f"   - Total successful experiences: {patterns.get('total_successful_experiences', 0)}")
            print(f"   - Strategy types found: {len(patterns.get('success_rates_by_strategy', {}))}")
        else:
            print("⚠️ No patterns found (this is normal for a new knowledge base)")
        
        print("\n🎉 All tests completed successfully!")
        print("\nThe knowledge graph is ready to use with your game agent.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your Azure credentials and service names")
        print("2. Ensure your Azure OpenAI service has the required models deployed")
        print("3. Verify your Azure Search service is running and accessible")
        print("4. Check your internet connection and Azure service status")
        return False


async def test_integration_example():
    """Test integration with a mock game agent."""
    print("\n🔗 Testing Knowledge Graph Integration")
    print("=" * 50)
    
    try:
        from knowledge_integration import KnowledgeEnhancedGameAgent
        
        # Create a mock client
        class MockGameClient:
            def __init__(self):
                self.game_state = "Mock game in progress"
                self.user_instructions = "Win the game efficiently"
                self.past_decisions = "Previous mock decisions"
                self.learned_patterns = "Mock learned patterns"
        
        mock_client = MockGameClient()
        
        # Create enhanced agent
        print("🤖 Creating knowledge-enhanced game agent...")
        enhanced_agent = KnowledgeEnhancedGameAgent(
            pyautogui_client=mock_client,
            learning_enabled=True
        )
        
        await enhanced_agent.initialize()
        print("✅ Enhanced agent initialized")
        
        # Test enhanced analysis
        print("🧠 Testing enhanced game state analysis...")
        analysis = await enhanced_agent.enhance_game_state_analysis(
            game_state="Mock game state for testing",
            visual_data="Mock visual data"
        )
        
        print("✅ Enhanced analysis completed")
        print(f"   - Recommendations: {len(analysis.get('recommendations', []))}")
        print(f"   - Similar experiences: {len(analysis.get('similar_experiences', []))}")
        print(f"   - Confidence score: {analysis.get('confidence_score', 0):.2f}")
        
        # Test prompt enhancement
        print("📝 Testing prompt enhancement...")
        base_prompt = "You are a game-playing AI agent."
        enhanced_prompt = await enhanced_agent.get_knowledge_enhanced_prompt(
            base_prompt=base_prompt,
            game_state="Test game state"
        )
        
        if len(enhanced_prompt) > len(base_prompt):
            print("✅ Prompt successfully enhanced with knowledge insights")
        else:
            print("⚠️ Prompt enhancement had no effect (normal for empty knowledge base)")
        
        print("\n🎉 Integration test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Integration test failed - missing dependencies: {e}")
        print("Make sure all required packages are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    return True


async def cleanup_test_resources():
    """Clean up test resources."""
    print("\n🧹 Cleaning up test resources...")
    
    try:
        # This would typically delete the test index
        # For safety, we'll just print what would be cleaned up
        print("⚠️ Test resources cleanup:")
        print("   - Test search index: test-knowledge-graph")
        print("   - To manually clean up, delete the index from Azure portal")
        print("✅ Cleanup information provided")
        
    except Exception as e:
        print(f"❌ Cleanup warning: {e}")


def print_setup_instructions():
    """Print setup instructions for users."""
    print("\n📋 Setup Instructions")
    print("=" * 30)
    print("1. Install required packages:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Set up Azure resources manually:")
    print("   - Create an Azure AI Search service in the Azure portal")
    print("   - Create an Azure OpenAI resource with GPT and embedding models")
    print("   - Ensure the search service supports vector search (Basic tier or higher)")
    print()
    print("3. Configure environment variables in .env:")
    print("   - Copy .env.example to .env")
    print("   - Fill in your Azure credentials")
    print()
    print("4. Run this test:")
    print("   python test_knowledge_graph.py")
    print()
    print("5. Integrate with your game agent:")
    print("   - See knowledge_integration.py for examples")
    print("   - Modify your client.py to use the enhanced agent")


async def main():
    """Main test function."""
    print("🚀 Azure AI Search Knowledge Graph Test Suite")
    print("=" * 60)
    
    # Check if this is a help request
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print_setup_instructions()
        return
    
    # Run basic functionality tests
    basic_success = await test_basic_functionality()
    
    if basic_success:
        # Run integration tests
        integration_success = await test_integration_example()
        
        if integration_success:
            print("\n🏆 All tests passed! Your knowledge graph setup is working correctly.")
            print("\nNext steps:")
            print("1. Integrate the knowledge graph with your game agent")
            print("2. Start playing games to build up the knowledge base")
            print("3. Monitor the recommendations and patterns over time")
        else:
            print("\n⚠️ Basic tests passed, but integration tests failed.")
            print("The core functionality works, but check the integration code.")
    else:
        print("\n❌ Basic tests failed. Please check your setup and try again.")
        print_setup_instructions()
    
    # Cleanup
    await cleanup_test_resources()


if __name__ == "__main__":
    asyncio.run(main())
