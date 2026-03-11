"""
Application routes registration module

This module centralizes all route imports and registrations.
It acts as a single entry point for managing all API routes.
"""

from fastapi import FastAPI, APIRouter, HTTPException
from app.schemas import (
    TransportPredictionRequest,
    TransportPredictionResponse,
    BatchPredictionResponse,
)
from app.schemas.knowledge import (
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    KnowledgeStatsResponse,
)
from app.controllers.prediction import (
    predict_best_transport,
    batch_predictions,
    health_check,
    model_info,
)
from app.controllers.knowledge import (
    search_knowledge,
    upload_documents,
    get_knowledge_stats,
    knowledge_health,
    list_collections,
    reset_collection,
    get_document,
    delete_document,
)

# Create router for transport routes
transport_router = APIRouter(prefix="/api/transport", tags=["transport"])


@transport_router.post("/predict", response_model=TransportPredictionResponse)
async def predict_route(request: TransportPredictionRequest):
    """
    Get the best transport recommendation for a single trip
    
    Returns:
        - prediction: Recommended transport type
        - confidence: Confidence score (0-1)
        - all_scores: Scores for all transport types
    """
    try:
        return await predict_best_transport(request)
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Models not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@transport_router.post("/batch", response_model=BatchPredictionResponse)
async def batch_route(requests: list[TransportPredictionRequest]):
    """
    Get transport recommendations for multiple trips in one request
    
    Args:
        requests: List of TransportPredictionRequest objects
        
    Returns:
        - status: "success" or "partial"
        - total: Number of predictions processed
        - predictions: List of prediction results
    """
    try:
        return await batch_predictions(requests)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@transport_router.get("/health")
async def health_route():
    """Health check endpoint"""
    return await health_check()


@transport_router.get("/models/info")
async def model_info_route():
    """Get information about loaded models"""
    try:
        return await model_info()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Create router for knowledge/RAG routes
knowledge_router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@knowledge_router.post("/search", response_model=KnowledgeSearchResponse)
async def search_route(request: KnowledgeSearchRequest):
    """
    Semantic search with optional RAG answer generation
    
    Returns relevant documents and optionally generates a natural language answer.
    
    Args:
        - query: User's question
        - top_k: Number of results (default: 5)
        - collection_name: Collection to search (optional)
        - use_rag: Generate answer using LLM (default: True)
        - language: auto/en/si/ta
    """
    return await search_knowledge(request)


@knowledge_router.post("/upload", response_model=DocumentUploadResponse)
async def upload_route(request: DocumentUploadRequest):
    """
    Upload knowledge documents to vector database
    
    Args:
        - documents: List of KnowledgeDocument objects
        - collection_name: Target collection
        - overwrite: Replace existing documents
    """
    return await upload_documents(request)


@knowledge_router.get("/stats", response_model=KnowledgeStatsResponse)
async def stats_route():
    """
    Get knowledge base statistics
    
    Returns:
        - total_documents: Total documents across all collections
        - collections: Stats per collection
        - embedding_model: Model used for embeddings
    """
    return await get_knowledge_stats()


@knowledge_router.get("/health")
async def knowledge_health_route():
    """Health check for RAG system"""
    return await knowledge_health()


@knowledge_router.get("/collections")
async def collections_route():
    """List all collections"""
    return {"collections": await list_collections()}


@knowledge_router.delete("/collections/{collection_name}")
async def reset_collection_route(collection_name: str):
    """Reset (delete and recreate) a collection"""
    return await reset_collection(collection_name)


@knowledge_router.get("/documents/{doc_id}")
async def get_document_route(doc_id: str, collection_name: str = "bus_fares"):
    """Get a specific document by ID"""
    return await get_document(doc_id, collection_name)


@knowledge_router.delete("/documents/{doc_id}")
async def delete_document_route(doc_id: str, collection_name: str = "bus_fares"):
    """Delete a document by ID"""
    return await delete_document(doc_id, collection_name)


def register_routes(app: FastAPI) -> None:
    """
    Register all application routes
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(transport_router)
    app.include_router(knowledge_router)
