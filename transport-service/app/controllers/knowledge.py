"""
Knowledge Base API Controller

Handles HTTP endpoints for RAG knowledge base operations.
Includes search, document upload, and statistics.
"""

import logging
from typing import List
from fastapi import HTTPException

from app.schemas.knowledge import (
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    KnowledgeStatsResponse,
    KnowledgeDocument,
    CollectionStats
)
from app.services.rag_service import get_rag_service
from app.services.vector_db_service import get_vector_db_service

logger = logging.getLogger(__name__)


# ================================================
# Search Endpoint
# ================================================

async def search_knowledge(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """
    Semantic search with optional RAG answer generation
    
    Args:
        request: Search parameters
        
    Returns:
        Search results with optional generated answer
    """
    try:
        rag_service = get_rag_service()
        response = rag_service.search_knowledge(request)
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ================================================
# Document Upload Endpoint
# ================================================

async def upload_documents(request: DocumentUploadRequest) -> DocumentUploadResponse:
    """
    Upload knowledge documents to vector database
    
    Args:
        request: Documents to upload
        
    Returns:
        Upload status and statistics
    """
    try:
        vector_db = get_vector_db_service()
        
        documents_added = 0
        duplicates_skipped = 0
        failed_ids = []
        
        # Check if overwrite mode
        if request.overwrite:
            # Delete existing documents first
            for doc in request.documents:
                try:
                    vector_db.delete_document(doc.id, request.collection_name)
                except Exception:
                    pass  # Document might not exist
        
        # Add documents with deduplication
        result = vector_db.add_documents(
            documents=request.documents,
            collection_name=request.collection_name,
            skip_duplicates=True  # Always skip duplicates by default
        )
        
        if result.get("success"):
            documents_added = result.get("documents_added", 0)
            duplicates_skipped = result.get("duplicates_skipped", 0)
            
            response_msg = f"Added {documents_added} documents"
            if duplicates_skipped > 0:
                response_msg += f" (skipped {duplicates_skipped} duplicate routes)"
            response_msg += f" to {request.collection_name}"
            
            return DocumentUploadResponse(
                success=True,
                documents_added=documents_added,
                documents_updated=0,
                duplicates_skipped=duplicates_skipped,
                failed_ids=failed_ids,
                message=response_msg
            )
        else:
            raise Exception(result.get("error", "Unknown error"))
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


# ================================================
# Statistics Endpoint
# ================================================

async def get_knowledge_stats() -> KnowledgeStatsResponse:
    """
    Get knowledge base statistics
    
    Returns:
        Statistics for all collections
    """
    try:
        vector_db = get_vector_db_service()
        
        # Get all collections
        collection_names = vector_db.list_collections()
        
        # Get stats for each collection
        collections_stats = []
        total_docs = 0
        
        for name in collection_names:
            stats = vector_db.get_collection_stats(name)
            collections_stats.append(stats)
            total_docs += stats.document_count
        
        return KnowledgeStatsResponse(
            total_documents=total_docs,
            collections=collections_stats,
            embedding_model=vector_db.embedding_model_name,
            vector_db_type="chromadb"
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


# ================================================
# Health Check Endpoint
# ================================================

async def knowledge_health() -> dict:
    """
    Health check for RAG system
    
    Returns:
        Health status of components
    """
    try:
        rag_service = get_rag_service()
        health = rag_service.health_check()
        
        return health
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "rag_service": "unhealthy",
            "error": str(e)
        }


# ================================================
# Collection Management Endpoints
# ================================================

async def list_collections() -> List[str]:
    """
    List all available collections
    
    Returns:
        List of collection names
    """
    try:
        vector_db = get_vector_db_service()
        collections = vector_db.list_collections()
        
        return collections
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}"
        )


async def reset_collection(collection_name: str) -> dict:
    """
    Delete and recreate a collection
    
    Args:
        collection_name: Name of collection to reset
        
    Returns:
        Status message
    """
    try:
        vector_db = get_vector_db_service()
        success = vector_db.reset_collection(collection_name)
        
        if success:
            return {
                "success": True,
                "message": f"Collection '{collection_name}' has been reset"
            }
        else:
            raise Exception("Reset operation failed")
        
    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset collection: {str(e)}"
        )


async def get_document(doc_id: str, collection_name: str = "bus_fares") -> KnowledgeDocument:
    """
    Retrieve a specific document by ID
    
    Args:
        doc_id: Document identifier
        collection_name: Collection name
        
    Returns:
        Knowledge document
    """
    try:
        vector_db = get_vector_db_service()
        document = vector_db.get_document(doc_id, collection_name)
        
        if document:
            return document
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{doc_id}' not found in collection '{collection_name}'"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document: {str(e)}"
        )


async def delete_document(doc_id: str, collection_name: str = "bus_fares") -> dict:
    """
    Delete a document by ID
    
    Args:
        doc_id: Document identifier
        collection_name: Collection name
        
    Returns:
        Status message
    """
    try:
        vector_db = get_vector_db_service()
        success = vector_db.delete_document(doc_id, collection_name)
        
        if success:
            return {
                "success": True,
                "message": f"Document '{doc_id}' deleted from '{collection_name}'"
            }
        else:
            raise Exception("Delete operation failed")
        
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )
