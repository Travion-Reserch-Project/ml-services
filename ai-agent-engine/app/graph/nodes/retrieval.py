"""
Retrieval Node: Vector Database Search for Tourism Knowledge.

This node fetches relevant documents from ChromaDB based on the user's query.
It implements semantic search with metadata filtering for location-specific
and aspect-specific retrieval.

Research Pattern:
    Hybrid Retrieval - Combines vector similarity with metadata filtering
    to achieve both semantic relevance and contextual precision.

ChromaDB Structure:
    Collection: tourism_knowledge
    Documents: 480 (80 locations × 6 aspects)
    Aspects: _history, _adventure, _nature, _culture, _logistics, _vibe
    Embedding: text-embedding-3-small (1536 dimensions)
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from ..state import GraphState, RetrievedDocument

logger = logging.getLogger(__name__)

# Default path to vector database
DEFAULT_VECTOR_DB_PATH = Path(__file__).parent.parent.parent.parent / "vector_db"

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available")


class VectorDBService:
    """
    ChromaDB interface for tourism knowledge retrieval.

    This service manages connections to the ChromaDB vector store
    and provides semantic search with metadata filtering.

    Attributes:
        client: ChromaDB persistent client
        collection: Tourism knowledge collection
        embedding_function: Sentence transformer for queries
    """

    # Aspect weights for different query types
    ASPECT_WEIGHTS = {
        "history": ["_history", "_culture"],
        "adventure": ["_adventure", "_nature"],
        "nature": ["_nature", "_adventure"],
        "religious": ["_culture", "_history"],
        "logistics": ["_logistics", "_vibe"],
        "general": ["_vibe", "_logistics", "_history"]
    }

    def __init__(
        self,
        persist_dir: str = "./vector_db",
        collection_name: str = "tourism_knowledge"
    ):
        """
        Initialize VectorDB Service.

        Args:
            persist_dir: Path to ChromaDB persistent storage
            collection_name: Name of the collection to use
        """
        self.client = None
        self.collection = None
        self.enabled = False

        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=persist_dir)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": "Sri Lankan tourism knowledge base"}
                )
                self.enabled = True
                logger.info(f"VectorDB connected: {self.collection.count()} documents")
            except Exception as e:
                logger.error(f"Failed to connect to ChromaDB: {e}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        location_filter: Optional[str] = None,
        aspect_filter: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            n_results: Number of results to return
            location_filter: Filter by location name
            aspect_filter: Filter by aspect types

        Returns:
            List of RetrievedDocument objects

        Example:
            >>> service = VectorDBService()
            >>> docs = service.search(
            ...     "historical significance of Sigiriya",
            ...     location_filter="Sigiriya"
            ... )
        """
        if not self.enabled:
            return self._fallback_search(query)

        try:
            # Build where clause for filtering
            where_clause = None
            if location_filter or aspect_filter:
                conditions = []
                if location_filter:
                    conditions.append({"location": {"$eq": location_filter}})
                if aspect_filter:
                    conditions.append({"aspect": {"$in": aspect_filter}})

                if len(conditions) == 1:
                    where_clause = conditions[0]
                else:
                    where_clause = {"$and": conditions}

            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            # Convert to RetrievedDocument format
            documents = []
            for i in range(len(results["documents"][0])):
                doc = RetrievedDocument(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    relevance_score=1 - results["distances"][0][i],  # Convert distance to similarity
                    source="chromadb"
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return self._fallback_search(query)

    def search_by_location(
        self,
        location_name: str,
        aspects: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """
        Get all documents for a specific location.

        Args:
            location_name: Name of the location
            aspects: Specific aspects to retrieve

        Returns:
            List of documents for the location
        """
        return self.search(
            query=location_name,
            n_results=10,
            location_filter=location_name,
            aspect_filter=aspects
        )

    def _fallback_search(self, query: str) -> List[RetrievedDocument]:
        """
        Fallback search when ChromaDB is unavailable.

        Returns mock documents for testing/development.
        """
        logger.warning("Using fallback search (mock data)")

        # Return some mock documents
        mock_docs = [
            RetrievedDocument(
                content=f"Sri Lanka is a beautiful island nation with rich cultural heritage. "
                        f"Your query about '{query}' relates to our extensive tourism offerings.",
                metadata={"location": "General", "aspect": "_vibe"},
                relevance_score=0.75,
                source="fallback"
            ),
            RetrievedDocument(
                content="Popular destinations include Sigiriya Rock Fortress, Galle Fort, "
                        "Temple of the Tooth in Kandy, and the beaches of the southern coast.",
                metadata={"location": "General", "aspect": "_logistics"},
                relevance_score=0.70,
                source="fallback"
            )
        ]

        return mock_docs


# Global VectorDB instance
_vectordb_service: Optional[VectorDBService] = None


def get_vectordb_service(
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None
) -> VectorDBService:
    """Get or create VectorDB service singleton."""
    global _vectordb_service
    if _vectordb_service is None:
        _vectordb_service = VectorDBService(
            persist_dir=persist_dir or str(DEFAULT_VECTOR_DB_PATH),
            collection_name=collection_name or "tourism_knowledge"
        )
    return _vectordb_service


async def retrieval_node(
    state: GraphState,
    vectordb: Optional[VectorDBService] = None
) -> GraphState:
    """
    Retrieval Node: Fetch relevant context from ChromaDB.

    This node performs semantic search to find documents relevant to
    the user's query. It uses extracted entities to apply metadata
    filters for more precise retrieval.

    Args:
        state: Current graph state
        vectordb: Optional VectorDB service instance

    Returns:
        Updated GraphState with retrieved documents

    Research Note:
        The retrieval implements a "Retrieve-then-Rerank" pattern where
        we over-fetch candidates and let the Grader node filter them.
    """
    query = state["user_query"]
    target_location = state.get("target_location")

    logger.info(f"Retrieval for: {query[:50]}... (location: {target_location})")

    # Get or create VectorDB service
    if vectordb is None:
        vectordb = get_vectordb_service()

    # Determine search strategy based on intent
    intent = state.get("intent")
    n_results = 7  # Over-fetch for reranking

    # Apply location filter if we have a target
    location_filter = target_location if target_location else None

    # Perform search
    documents = vectordb.search(
        query=query,
        n_results=n_results,
        location_filter=location_filter
    )

    # Log retrieval metrics
    logger.info(f"Retrieved {len(documents)} documents")
    for i, doc in enumerate(documents[:3]):
        logger.debug(f"  Doc {i+1}: {doc['metadata'].get('location', 'N/A')} "
                    f"({doc['relevance_score']:.3f})")

    # Update state
    return {
        **state,
        "retrieved_documents": documents,
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [{
            "timestamp": datetime.now().isoformat(),
            "check_type": "retrieval",
            "input_context": {
                "query": query,
                "location_filter": location_filter
            },
            "result": "success" if documents else "empty",
            "details": f"Retrieved {len(documents)} documents",
            "action_taken": None
        }]
    }
