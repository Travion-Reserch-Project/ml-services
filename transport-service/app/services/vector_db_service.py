"""
Vector Database Service

Handles ChromaDB operations for knowledge storage and retrieval.
Supports semantic search with multilingual embeddings.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from datetime import datetime

from app.schemas.knowledge import (
    KnowledgeDocument,
    SearchResult,
    DocumentMetadata,
    CollectionStats
)

logger = logging.getLogger(__name__)


class VectorDBService:
    """
    Service for managing vector database operations
    
    Features:
    - Document embedding and storage
    - Semantic search
    - Multiple collection support
    - Batch operations
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize Vector DB service
        
        Args:
            persist_dir: Directory for ChromaDB storage
            embedding_model: Sentence transformer model name
        """
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR",
            "./vector_db/chroma_data"
        )
        
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        logger.info(f"✓ Vector DB initialized at {self.persist_dir}")
    
    def _get_or_create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(
                name=collection_name
            )
            logger.debug(f"Using existing collection: {collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata or {"created_at": datetime.now().isoformat()}
            )
            logger.info(f"✓ Created new collection: {collection_name}")
        
        return collection
    
    def add_documents(
        self,
        documents: List[KnowledgeDocument],
        collection_name: str = "bus_fares",
        skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Add documents to vector database
        
        Args:
            documents: List of knowledge documents
            collection_name: Target collection
            skip_duplicates: Skip documents that already exist (by ID)
            
        Returns:
            Dictionary with operation results
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            
            # Get existing document IDs for deduplication
            existing_ids = set()
            if skip_duplicates:
                try:
                    existing_data = collection.get()
                    if existing_data and existing_data.get("ids"):
                        existing_ids = set(existing_data["ids"])
                    logger.info(f"Found {len(existing_ids)} existing documents in {collection_name}")
                except Exception as e:
                    logger.warning(f"Could not retrieve existing documents: {e}")
            
            # Filter out duplicates
            docs_to_add = []
            duplicates_skipped = 0
            
            for doc in documents:
                if skip_duplicates and doc.id in existing_ids:
                    logger.debug(f"Skipping duplicate: {doc.id}")
                    duplicates_skipped += 1
                    continue
                
                docs_to_add.append(doc)
                existing_ids.add(doc.id)  # Mark as added
            
            if not docs_to_add:
                logger.warning(f"No new documents to add (all {len(documents)} were duplicates)")
                return {
                    "success": True,
                    "added": 0,
                    "duplicates_skipped": duplicates_skipped,
                    "total": len(existing_ids),
                    "collection": collection_name,
                    "message": "All documents were duplicates and skipped"
                }
            
            # Prepare data for ChromaDB
            ids = [doc.id for doc in docs_to_add]
            texts = [doc.content for doc in docs_to_add]
            # Filter metadata - ChromaDB only accepts str, int, float, bool
            # Remove None values and non-primitive types like lists
            metadatas = [
                {
                    k: v for k, v in doc.metadata.model_dump().items() 
                    if v is not None and isinstance(v, (str, int, float, bool))
                }
                for doc in docs_to_add
            ]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(docs_to_add)} documents...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            embeddings_list = embeddings.tolist()
            
            # Add to collection
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            
            logger.info(f"✓ Added {len(docs_to_add)} documents to {collection_name}")
            
            return {
                "success": True,
                "added": len(docs_to_add),
                "total": len(existing_ids),
                "duplicates_skipped": duplicates_skipped,
                "collection": collection_name,
                "message": f"Added {len(docs_to_add)} documents (skipped {duplicates_skipped} duplicates)"
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return {
                "success": False,
                "added": 0,
                "error": str(e)
            }
    
    def search(
        self,
        query: str,
        collection_name: str = "bus_fares",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search for relevant documents
        
        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of search results
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )
            
            # Parse results
            search_results = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    # Calculate similarity score (1 - distance)
                    distance = results['distances'][0][i] if results.get('distances') else 0
                    score = 1 - distance
                    
                    metadata_dict = results['metadatas'][0][i]
                    
                    search_results.append(SearchResult(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        score=score,
                        distance=distance,
                        metadata=DocumentMetadata(**metadata_dict)
                    ))
            
            logger.info(f"✓ Found {len(search_results)} results for query: {query[:50]}...")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def update_document(
        self,
        doc_id: str,
        document: KnowledgeDocument,
        collection_name: str = "bus_fares"
    ) -> bool:
        """
        Update existing document
        
        Args:
            doc_id: Document ID
            document: Updated document
            collection_name: Collection name
            
        Returns:
            Success status
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            
            # Generate embedding
            embedding = self.embedding_model.encode([document.content])[0].tolist()
            
            # Update
            collection.update(
                ids=[doc_id],
                documents=[document.content],
                embeddings=[embedding],
                metadatas=[document.metadata.model_dump()]
            )
            
            logger.info(f"✓ Updated document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    def delete_document(
        self,
        doc_id: str,
        collection_name: str = "bus_fares"
    ) -> bool:
        """
        Delete document by ID
        
        Args:
            doc_id: Document ID
            collection_name: Collection name
            
        Returns:
            Success status
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            collection.delete(ids=[doc_id])
            
            logger.info(f"✓ Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_collection_stats(
        self,
        collection_name: str = "bus_fares"
    ) -> CollectionStats:
        """
        Get statistics for a collection
        
        Args:
            collection_name: Collection name
            
        Returns:
            Collection statistics
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            count = collection.count()
            
            return CollectionStats(
                name=collection_name,
                document_count=count,
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {e}")
            return CollectionStats(
                name=collection_name,
                document_count=0
            )
    
    def list_collections(self) -> List[str]:
        """Get list of all collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def reset_collection(self, collection_name: str) -> bool:
        """
        Delete and recreate collection
        
        Args:
            collection_name: Collection to reset
            
        Returns:
            Success status
        """
        try:
            self.client.delete_collection(name=collection_name)
            self._get_or_create_collection(collection_name)
            logger.info(f"✓ Reset collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection {collection_name}: {e}")
            return False
    
    def get_document(
        self,
        doc_id: str,
        collection_name: str = "bus_fares"
    ) -> Optional[KnowledgeDocument]:
        """
        Retrieve document by ID
        
        Args:
            doc_id: Document ID
            collection_name: Collection name
            
        Returns:
            Knowledge document or None
        """
        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(ids=[doc_id])
            
            if results['ids'] and len(results['ids']) > 0:
                return KnowledgeDocument(
                    id=results['ids'][0],
                    content=results['documents'][0],
                    type="general",
                    metadata=DocumentMetadata(**results['metadatas'][0])
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None


# ================================================
# Singleton Instance
# ================================================

_vector_db_service = None


def get_vector_db_service() -> VectorDBService:
    """Get or create singleton instance of VectorDBService"""
    global _vector_db_service
    
    if _vector_db_service is None:
        _vector_db_service = VectorDBService()
    
    return _vector_db_service


# ================================================
# Testing
# ================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the service
    service = VectorDBService()
    
    # Create test documents
    test_docs = [
        KnowledgeDocument(
            id="test_001",
            type="bus_fare",
            content="Bus fare from Colombo to Galle is LKR 250 for normal service.",
            metadata=DocumentMetadata(
                origin="Colombo",
                destination="Galle",
                fare_normal=250.0,
                distance_km=119,
                language="en"
            )
        ),
        KnowledgeDocument(
            id="test_002",
            type="bus_fare",
            content="කොළඹ සිට ගාල්ල බස් ගාස්තුව සාමාන්‍ය සේවාව සඳහා රුපියල් 250කි.",
            metadata=DocumentMetadata(
                origin="Colombo",
                destination="Galle",
                fare_normal=250.0,
                distance_km=119,
                language="si"
            )
        )
    ]
    
    # Add documents
    result = service.add_documents(test_docs, "test_collection")
    print(f"\n✓ Add result: {result}")
    
    # Search
    results = service.search("Colombo to Galle fare", "test_collection", top_k=2)
    print(f"\n✓ Search results:")
    for r in results:
        print(f"  - {r.content[:50]}... (score: {r.score:.3f})")
    
    # Get stats
    stats = service.get_collection_stats("test_collection")
    print(f"\n✓ Collection stats: {stats.document_count} documents")
