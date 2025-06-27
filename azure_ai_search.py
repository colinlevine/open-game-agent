"""
Azure AI Search Integration

This module provides integration with Azure AI Search for building and querying
vector databases that can enhance the game agent's decision-making capabilities.
It supports creating semantic knowledge bases from game data and strategies.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import networkx as nx
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    ComplexField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameKnowledgeNode:
    """Represents a node in the game knowledge graph."""
    id: str
    type: str  # e.g., "strategy", "pattern", "outcome", "state"
    content: str
    properties: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            from datetime import timezone
            self.created_at = datetime.now(timezone.utc)


@dataclass
class GameKnowledgeEdge:
    """Represents an edge/relationship in the game knowledge graph."""
    source_id: str
    target_id: str
    relationship_type: str  # e.g., "leads_to", "similar_to", "causes", "requires"
    weight: float = 1.0
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class AzureAISearchKnowledgeGraph:
    """
    Azure AI Search integration for game knowledge graphs.
    
    This class manages a knowledge graph stored in Azure AI Search that can help
    the game agent make better decisions by learning from past experiences and
    storing strategic knowledge.
    """
    
    def __init__(self, 
                 search_service_name: str = None,
                 search_api_key: str = None,
                 index_name: str = "game-knowledge-graph",
                 use_managed_identity: bool = False):
        """
        Initialize the Azure AI Search Knowledge Graph.
        
        Args:
            search_service_name: Azure Search service name
            search_api_key: Azure Search API key (if not using managed identity)
            index_name: Name of the search index for the knowledge graph
            use_managed_identity: Whether to use Azure managed identity for authentication
        """
        # Get configuration from environment if not provided
        self.search_service_name = search_service_name or os.getenv("AZURE_SEARCH_SERVICE_NAME")
        self.search_api_key = search_api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = index_name
        self.use_managed_identity = use_managed_identity
        
        if not self.search_service_name:
            raise ValueError("Azure Search service name is required")
        
        # Set up Azure Search endpoint
        self.search_endpoint = f"https://{self.search_service_name}.search.windows.net"
        
        # Set up authentication
        if self.use_managed_identity:
            self.credential = DefaultAzureCredential()
        else:
            if not self.search_api_key:
                raise ValueError("Azure Search API key is required when not using managed identity")
            self.credential = AzureKeyCredential(self.search_api_key)
        
        # Initialize Azure Search clients
        self.index_client = SearchIndexClient(
            endpoint=self.search_endpoint,
            credential=self.credential
        )
        
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # Initialize Azure OpenAI for embeddings
        self.azure_openai = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # In-memory NetworkX graph for complex graph operations
        self.graph = nx.DiGraph()
        
    async def initialize_index(self):
        """Create the Azure Search index if it doesn't exist."""
        try:
            # Check if index already exists
            try:
                existing_index = self.index_client.get_index(self.index_name)
                logger.info(f"Index '{self.index_name}' already exists")
                return
            except Exception:
                logger.info(f"Index '{self.index_name}' does not exist, creating...")
            
            # Define the search index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="node_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,  # OpenAI text-embedding-ada-002 dimension
                    vector_search_profile_name="game-knowledge-profile"
                ),
                SimpleField(name="game_state", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="strategy_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="success_rate", type=SearchFieldDataType.Double, filterable=True),
                SimpleField(name="difficulty_level", type=SearchFieldDataType.String, filterable=True),
                SearchField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SimpleField(name="updated_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True)
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="game-knowledge-hnsw",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="game-knowledge-profile",
                        algorithm_configuration_name="game-knowledge-hnsw"
                    )
                ]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            result = self.index_client.create_index(index)
            logger.info(f"Created search index: {result.name}")
            
        except Exception as e:
            logger.error(f"Error creating search index: {e}")
            raise
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text using Azure OpenAI."""
        try:
            # Use the embedding deployment name from environment variables
            embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            
            response = self.azure_openai.embeddings.create(
                model=embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    async def add_knowledge_node(self, node: GameKnowledgeNode) -> bool:
        """Add a knowledge node to the graph and search index."""
        try:
            # Generate embeddings if not provided
            if not node.embeddings:
                node.embeddings = await self.generate_embeddings(node.content)
            
            # Add to NetworkX graph
            self.graph.add_node(
                node.id,
                type=node.type,
                content=node.content,
                properties=node.properties,
                created_at=node.created_at
            )
            
            # Prepare document for Azure Search with flattened structure
            from datetime import timezone
            search_doc = {
                "@search.action": "upload",
                "id": node.id,
                "node_type": node.type,
                "content": node.content,
                "content_vector": node.embeddings,
                "game_state": str(node.properties.get("game_state", "")),
                "strategy_type": str(node.properties.get("strategy_type", "")),
                "success_rate": float(node.properties.get("success_rate", 0.0)),
                "difficulty_level": str(node.properties.get("difficulty_level", "")),
                "tags": list(node.properties.get("tags", [])),
                "created_at": node.created_at.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Index the document using upload_documents
            result = self.search_client.upload_documents([search_doc])
            
            if result[0].succeeded:
                logger.info(f"Successfully added knowledge node: {node.id}")
                return True
            else:
                logger.error(f"Failed to add knowledge node: {result[0].error_message}")
                return False
        except Exception as e:
            logger.error(f"Error adding knowledge node: {e}")
            return False
    
    async def add_knowledge_edge(self, edge: GameKnowledgeEdge) -> bool:
        """Add a knowledge edge/relationship to the graph."""
        try:
            # Add to NetworkX graph
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                relationship_type=edge.relationship_type,
                weight=edge.weight,
                properties=edge.properties
            )
            
            # Update the source node's relationships in Azure Search
            source_doc = self.search_client.get_document(key=edge.source_id)
            if not source_doc.get("relationships"):
                source_doc["relationships"] = []
            
            # Add the new relationship
            relationship = {
                "target_id": edge.target_id,
                "relationship_type": edge.relationship_type,
                "weight": edge.weight
            }
            
            source_doc["relationships"] .append(relationship)
            source_doc["updated_at"] = datetime.utcnow().isoformat()
            
            # Index update using merge_or_upload_documents
            result = self.search_client.merge_or_upload_documents([source_doc])
            
            if result[0].succeeded:
                logger.info(f"Successfully added knowledge edge: {edge.source_id} -> {edge.target_id}")
                return True
            else:
                logger.error(f"Failed to add knowledge edge: {result[0].error_message}")
                return False
        except Exception as e:
            logger.error(f"Error adding knowledge edge: {e}")
            return False
    
    async def search_similar_knowledge(self, 
                                     query: str, 
                                     node_types: List[str] = None,
                                     top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar knowledge using vector similarity and semantic search."""
        try:
            # Generate query embeddings
            query_embeddings = await self.generate_embeddings(query)
            
            # Build search filter
            search_filter = None
            if node_types:
                filter_conditions = " or ".join([f"node_type eq '{nt}'" for nt in node_types])
                search_filter = f"({filter_conditions})"
            
            # Perform vector search
            vector_query = VectorizedQuery(
                vector=query_embeddings,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=search_filter,
                select=["id", "node_type", "content", "game_state", "strategy_type", "success_rate", "difficulty_level", "tags"],
                top=top_k
            )
            
            knowledge_results = []
            for result in results:
                # Rebuild properties structure from flattened fields
                properties = {
                    "game_state": result.get("game_state", ""),
                    "strategy_type": result.get("strategy_type", ""),
                    "success_rate": result.get("success_rate", 0.0),
                    "difficulty_level": result.get("difficulty_level", ""),
                    "tags": result.get("tags", [])
                }
                
                knowledge_results.append({
                    "id": result["id"],
                    "type": result["node_type"],
                    "content": result["content"],
                    "properties": properties,
                    "relationships": [],  # Simplified for now
                    "score": result.get("@search.score", 0),
                    "reranker_score": result.get("@search.reranker_score", 0)
                })
            
            return knowledge_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def get_related_knowledge(self, 
                                  node_id: str, 
                                  relationship_types: List[str] = None,
                                  max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get knowledge nodes related to a given node through graph traversal."""
        try:
            if node_id not in self.graph:
                logger.warning(f"Node {node_id} not found in graph")
                return []
            
            related_nodes = []
            visited = set()
            
            def traverse(current_id, depth):
                if depth > max_depth or current_id in visited:
                    return
                
                visited.add(current_id)
                
                # Get neighbors from NetworkX graph
                for neighbor_id in self.graph.neighbors(current_id):
                    edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                    
                    # Filter by relationship type if specified
                    if relationship_types:
                        if edge_data.get("relationship_type") not in relationship_types:
                            continue
                    
                    # Get full node data from Azure Search
                    try:
                        neighbor_doc = self.search_client.get_document(key=neighbor_id)
                        related_nodes.append({
                            "id": neighbor_id,
                            "type": neighbor_doc["node_type"],
                            "content": neighbor_doc["content"],
                            "properties": neighbor_doc.get("properties", {}),
                            "relationship": edge_data.get("relationship_type"),
                            "weight": edge_data.get("weight", 1.0),
                            "depth": depth
                        })
                        
                        # Continue traversal
                        traverse(neighbor_id, depth + 1)
                        
                    except Exception as e:
                        logger.warning(f"Could not retrieve node {neighbor_id}: {e}")
            
            traverse(node_id, 1)
            return related_nodes
            
        except Exception as e:
            logger.error(f"Error getting related knowledge: {e}")
            return []
    
    async def add_game_experience(self, 
                                game_state: str,
                                action_taken: str,
                                outcome: str,
                                success: bool,
                                strategy_type: str = "general") -> str:
        """Add a game experience as knowledge nodes and relationships."""
        try:
            base_id = str(uuid4())
            
            # Create state node
            state_node = GameKnowledgeNode(
                id=f"state_{base_id}",
                type="game_state",
                content=game_state,
                properties={
                    "game_state": game_state,
                    "strategy_type": strategy_type,
                    "success_rate": 0.0,  # Not applicable for states
                    "difficulty_level": "medium",
                    "tags": ["game_state", strategy_type]
                }
            )
            
            # Create action node
            action_node = GameKnowledgeNode(
                id=f"action_{base_id}",
                type="action",
                content=action_taken,
                properties={
                    "game_state": "",  # Empty for actions
                    "strategy_type": strategy_type,
                    "success_rate": 0.0,  # Not applicable for actions
                    "difficulty_level": "medium",
                    "tags": ["action", strategy_type]
                }
            )
            
            # Create outcome node
            outcome_node = GameKnowledgeNode(
                id=f"outcome_{base_id}",
                type="outcome",
                content=outcome,
                properties={
                    "game_state": "",  # Empty for outcomes
                    "strategy_type": strategy_type,
                    "success_rate": 1.0 if success else 0.0,
                    "difficulty_level": "medium",
                    "tags": ["outcome", "success" if success else "failure", strategy_type]
                }
            )
            
            # Add nodes
            await self.add_knowledge_node(state_node)
            await self.add_knowledge_node(action_node)
            await self.add_knowledge_node(outcome_node)
            
            # Add relationships
            state_to_action = GameKnowledgeEdge(
                source_id=state_node.id,
                target_id=action_node.id,
                relationship_type="leads_to",
                weight=1.0
            )
            
            action_to_outcome = GameKnowledgeEdge(
                source_id=action_node.id,
                target_id=outcome_node.id,
                relationship_type="results_in",
                weight=1.0 if success else 0.5
            )
            
            await self.add_knowledge_edge(state_to_action)
            await self.add_knowledge_edge(action_to_outcome)
            
            logger.info(f"Added game experience: {base_id}")
            return base_id
            
        except Exception as e:
            logger.error(f"Error adding game experience: {e}")
            return ""
    
    async def get_strategic_recommendations(self, 
                                         current_state: str,
                                         strategy_type: str = None) -> List[Dict[str, Any]]:
        """Get strategic recommendations based on current game state and past experiences."""
        try:
            # Search for similar game states
            similar_states = await self.search_similar_knowledge(
                query=current_state,
                node_types=["game_state"],
                top_k=10
            )
            
            recommendations = []
            
            for state in similar_states:
                # Get related actions and outcomes
                related_knowledge = await self.get_related_knowledge(
                    node_id=state["id"],
                    relationship_types=["leads_to", "results_in"],
                    max_depth=2
                )
                
                # Group by action-outcome pairs
                actions = [r for r in related_knowledge if r["type"] == "action"]
                
                for action in actions:
                    # Find outcomes for this action
                    action_outcomes = await self.get_related_knowledge(
                        node_id=action["id"],
                        relationship_types=["results_in"],
                        max_depth=1
                    )
                    
                    outcomes = [o for o in action_outcomes if o["type"] == "outcome"]
                    
                    if outcomes:
                        # Calculate success rate
                        successful_outcomes = [o for o in outcomes if o["properties"].get("success", False)]
                        success_rate = len(successful_outcomes) / len(outcomes) if outcomes else 0
                        
                        recommendations.append({
                            "action": action["content"],
                            "expected_outcome": outcomes[0]["content"] if outcomes else "Unknown",
                            "success_rate": success_rate,
                            "confidence": state["score"],
                            "similar_state": state["content"],
                            "strategy_type": action["properties"].get("strategy_type", "general")
                        })
            
            # Sort by success rate and confidence
            recommendations.sort(key=lambda x: (x["success_rate"], x["confidence"]), reverse=True)
            
            # Filter by strategy type if specified
            if strategy_type:
                recommendations = [r for r in recommendations if r["strategy_type"] == strategy_type]
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error getting strategic recommendations: {e}")
            return []
    
    async def analyze_patterns(self, strategy_type: str = None) -> Dict[str, Any]:
        """Analyze patterns in the knowledge graph to identify successful strategies."""
        try:
            # Build search filter
            search_filter = None
            if strategy_type:
                search_filter = f"strategy_type eq '{strategy_type}'"
            
            # Get all successful outcomes (success_rate > 0.5)
            successful_outcomes = self.search_client.search(
                search_text="*",
                filter=f"node_type eq 'outcome' and success_rate gt 0.5{' and ' + search_filter if search_filter else ''}",
                select=["id", "content", "game_state", "strategy_type", "success_rate", "difficulty_level", "tags"],
                top=1000
            )
            
            patterns = {
                "total_successful_experiences": 0,
                "common_successful_actions": {},
                "success_rates_by_strategy": {},
                "most_effective_patterns": []
            }
            
            for outcome in successful_outcomes:
                patterns["total_successful_experiences"] += 1
                
                # Analyze strategy type
                outcome_strategy = outcome.get("strategy_type", "general")
                if outcome_strategy not in patterns["success_rates_by_strategy"]:
                    patterns["success_rates_by_strategy"][outcome_strategy] = {"success": 0, "total": 0}
                patterns["success_rates_by_strategy"][outcome_strategy]["success"] += 1
                
                # Simplified: for now we'll skip the complex relationship analysis
                # In a real implementation, you'd store relationships in a separate index
                # or use a different approach
            
            # Calculate success rates
            for strategy in patterns["success_rates_by_strategy"]:
                # Get total count for this strategy
                total_outcomes = self.search_client.search(
                    search_text="*",
                    filter=f"node_type eq 'outcome' and strategy_type eq '{strategy}'",
                    top=1000,
                    include_total_count=True
                )
                
                total_count = total_outcomes.get_count() or 0
                patterns["success_rates_by_strategy"][strategy]["total"] = total_count
                
                if total_count > 0:
                    success_rate = patterns["success_rates_by_strategy"][strategy]["success"] / total_count
                    patterns["success_rates_by_strategy"][strategy]["rate"] = success_rate
            
            # Sort common successful actions
            sorted_actions = sorted(
                patterns["common_successful_actions"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            patterns["most_effective_patterns"] = sorted_actions[:10]
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {}
    
    async def export_graph_visualization(self, output_path: str = "knowledge_graph.html"):
        """Export the knowledge graph as an interactive HTML visualization."""
        try:
            from pyvis.network import Network
            
            # Create a pyvis network
            net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
            
            # Add nodes
            for node_id, node_data in self.graph.nodes(data=True):
                node_type = node_data.get("type", "unknown")
                content = node_data.get("content", "")[:50] + "..." if len(node_data.get("content", "")) > 50 else node_data.get("content", "")
                
                # Color by type
                color_map = {
                    "game_state": "#ff6b6b",
                    "action": "#4ecdc4",
                    "outcome": "#45b7d1",
                    "strategy": "#96ceb4",
                    "pattern": "#feca57"
                }
                
                net.add_node(
                    node_id,
                    label=f"{node_type}\n{content}",
                    color=color_map.get(node_type, "#ddd"),
                    title=node_data.get("content", "")
                )
            
            # Add edges
            for source, target, edge_data in self.graph.edges(data=True):
                relationship_type = edge_data.get("relationship_type", "related")
                weight = edge_data.get("weight", 1.0)
                
                net.add_edge(
                    source,
                    target,
                    label=relationship_type,
                    width=weight * 2,
                    title=f"{relationship_type} (weight: {weight})"
                )
            
            # Configure physics
            net.set_options("""
            {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 100}
              }
            }
            """)
            
            # Save to file
            net.save_graph(output_path)
            logger.info(f"Knowledge graph visualization saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting graph visualization: {e}")


# Example usage and testing functions
async def test_knowledge_graph():
    """Test the knowledge graph functionality."""
    # Initialize the knowledge graph
    kg = AzureAISearchKnowledgeGraph(
        index_name="test-game-knowledge"
    )
    
    try:
        # Initialize the index
        await kg.initialize_index()
        
        # Add some test game experiences
        await kg.add_game_experience(
            game_state="Wordle game started, first guess needed",
            action_taken="Type 'ADIEU' as first guess",
            outcome="Got 2 yellow letters (A, E) and 3 gray letters",
            success=False,
            strategy_type="vowel_heavy"
        )
        
        await kg.add_game_experience(
            game_state="Second guess needed, have A and E in wrong positions",
            action_taken="Type 'BRAVE' to test common letters",
            outcome="Got green B, yellow A, green V, yellow E, gray R",
            success=False,
            strategy_type="consonant_test"
        )
        
        await kg.add_game_experience(
            game_state="Know B_V_E pattern with A somewhere",
            action_taken="Type 'ABOVE' as final guess",
            outcome="All green - solved the puzzle!",
            success=True,
            strategy_type="pattern_completion"
        )
        
        # Get recommendations for a similar state
        recommendations = await kg.get_strategic_recommendations(
            current_state="Wordle game started, need first guess"
        )
        
        print("Strategic Recommendations:")
        for rec in recommendations:
            print(f"  Action: {rec['action']}")
            print(f"  Success Rate: {rec['success_rate']:.2f}")
            print(f"  Confidence: {rec['confidence']:.2f}")
            print()
        
        # Analyze patterns
        patterns = await kg.analyze_patterns()
        print("Pattern Analysis:")
        print(f"  Total Successful Experiences: {patterns.get('total_successful_experiences', 0)}")
        print("  Most Effective Actions:")
        for action, count in patterns.get('most_effective_patterns', [])[:3]:
            print(f"    {action}: {count} times")
        
        # Export visualization
        await kg.export_graph_visualization("test_ai_search.html")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_knowledge_graph())
