"""
Vision Retrieval Node: CLIP-based Image Search via ChromaDB.

This node handles multimodal image search queries:
  - Text-to-Image: User asks "show me photos of Sigiriya" → CLIP text embedding
    is compared against pre-computed image embeddings in the image_knowledge collection.
  - Image-to-Image: User uploads a photo → CLIP image embedding is generated and
    matched against the collection for visually similar destinations.

The node also validates uploaded images using the ImageValidator to ensure
only tourism-related photos are accepted.

ChromaDB Collection:
    Name: image_knowledge
    Embedding Model: openai/clip-vit-base-patch32 (512-dim)
    Embeddings: Pre-computed (passed via query_embeddings, not ChromaDB's default fn)
    Distance: Cosine similarity
"""

import logging
import time
from typing import Dict, List, Optional

# Tracing
try:
    from ...utils.tracing import trace_node
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def trace_node(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..state import GraphState, ImageSearchResult
from ...config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded service singletons
_image_vectordb_service = None


class ImageVectorDBService:
    """
    Service for querying the image_knowledge ChromaDB collection
    using CLIP embeddings.

    Unlike the text-based VectorDBService (which uses OpenAI embeddings
    and ChromaDB's built-in embedding function), this service:
    - Uses pre-computed 512-dim CLIP embeddings
    - Passes query_embeddings directly to ChromaDB (no embedding function)
    - Supports both text queries (via CLIP text encoder) and image queries
      (via CLIP image encoder)
    """

    def __init__(self):
        self._collection = None
        self._clip_service = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy-initialize ChromaDB collection and CLIP service."""
        if self._initialized:
            return

        try:
            import chromadb
            import os

            # Use IMAGE_CHROMA_PERSIST_DIR if set, else fall back to CHROMA_PERSIST_DIR
            chroma_dir = getattr(settings, "IMAGE_CHROMA_PERSIST_DIR", None) or settings.CHROMA_PERSIST_DIR

            # Resolve relative path against the ai-agent-engine project root
            if not os.path.isabs(chroma_dir):
                project_root = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                )
                chroma_dir = os.path.join(project_root, chroma_dir)

            logger.info(f"ImageVectorDBService: ChromaDB path resolved to {chroma_dir}")
            client = chromadb.PersistentClient(path=chroma_dir)
            self._collection = client.get_collection(
                name=settings.IMAGE_COLLECTION_NAME,
            )
            logger.info(
                f"ImageVectorDBService: Connected to '{settings.IMAGE_COLLECTION_NAME}' "
                f"({self._collection.count()} documents)"
            )
        except Exception as e:
            logger.error(f"Failed to connect to image_knowledge collection: {e}")
            raise RuntimeError(
                f"image_knowledge collection not found. "
                f"Run scripts/build_image_knowledge.py first. Error: {e}"
            ) from e

        try:
            from ...services.clip_embedding_service import get_clip_service
            self._clip_service = get_clip_service()
            logger.info("ImageVectorDBService: CLIP service loaded")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP service: {e}")
            raise RuntimeError(f"CLIP service initialization failed: {e}") from e

        self._initialized = True

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        location_filter: Optional[str] = None,
    ) -> List[ImageSearchResult]:
        """
        Text-to-Image search: encode query text with CLIP and search.

        Args:
            query: Natural language description (e.g., "sunset at Sigiriya").
            top_k: Number of results to return.
            location_filter: Optional location name to filter by.

        Returns:
            List of ImageSearchResult dicts sorted by similarity.
        """
        self._ensure_initialized()

        query_embedding = self._clip_service.embed_text(query)
        return self._query_collection(query_embedding, top_k, location_filter)

    def search_by_image_base64(
        self,
        base64_string: str,
        top_k: int = 5,
        location_filter: Optional[str] = None,
    ) -> List[ImageSearchResult]:
        """
        Image-to-Image search: encode uploaded image with CLIP and search.

        Args:
            base64_string: Base64-encoded image (JPEG/PNG/WebP).
            top_k: Number of results to return.
            location_filter: Optional location name to filter by.

        Returns:
            List of ImageSearchResult dicts sorted by similarity.
        """
        self._ensure_initialized()

        query_embedding = self._clip_service.embed_image_from_base64(base64_string)
        return self._query_collection(query_embedding, top_k, location_filter)

    def _query_collection(
        self,
        query_embedding: List[float],
        top_k: int,
        location_filter: Optional[str] = None,
    ) -> List[ImageSearchResult]:
        """
        Query ChromaDB image_knowledge collection with a pre-computed embedding.

        Args:
            query_embedding: 512-dim CLIP embedding vector.
            top_k: Number of results.
            location_filter: Optional location name exact match.

        Returns:
            List of ImageSearchResult dicts.
        """
        where_filter = None
        if location_filter:
            where_filter = {"location_name": {"$eq": location_filter}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
            where=where_filter,
        )

        image_results: List[ImageSearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return image_results

        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB cosine distance = 1 - similarity
            similarity = 1.0 - distance

            image_results.append({
                "image_id": results["ids"][0][i],
                "location_name": meta.get("location_name", ""),
                "description": meta.get("description", ""),
                "image_url": meta.get("image_url", ""),
                "file_path": meta.get("file_path", ""),
                "similarity_score": round(similarity, 4),
                "tags": meta.get("tags", ""),
                "coordinates": {
                    "lat": meta.get("lat", 0.0),
                    "lng": meta.get("lng", 0.0),
                } if meta.get("lat") else None,
            })

        return image_results


def get_image_vectordb_service() -> ImageVectorDBService:
    """Get or create the ImageVectorDBService singleton."""
    global _image_vectordb_service
    if _image_vectordb_service is None:
        _image_vectordb_service = ImageVectorDBService()
    return _image_vectordb_service


@trace_node("vision_retrieval")
async def vision_retrieval_node(state: GraphState) -> GraphState:
    """
    Vision Retrieval Node: Handle image search queries.

    Workflow:
    1. If user uploaded an image (uploaded_image_base64):
       a. Validate with ImageValidator (tourism content check)
       b. If valid → Image-to-Image CLIP search
       c. If invalid → Set rejection message, skip search
    2. If text query with image intent:
       a. Text-to-Image CLIP search using the user's query

    The results are stored in state["image_search_results"] for the
    Generator Node to incorporate into the response.

    Args:
        state: Current graph state.

    Returns:
        Updated GraphState with image search results.
    """
    _start = time.time()
    query = state["user_query"]
    uploaded_image = state.get("uploaded_image_base64")
    target_location = state.get("target_location")

    logger.info(
        f"Vision Retrieval: query='{query[:50]}...', "
        f"has_image={'yes' if uploaded_image else 'no'}, "
        f"location={target_location}"
    )

    image_results: List[ImageSearchResult] = []
    validated = None
    validation_message = None

    try:
        service = get_image_vectordb_service()
        top_k = settings.IMAGE_RETRIEVAL_TOP_K

        if uploaded_image:
            # --- Image-to-Image flow ---
            # Step 1: Validate the uploaded image
            try:
                from ...services.image_validator import get_image_validator

                validator = get_image_validator()
                validation = validator.validate_base64_image(uploaded_image)

                validated = validation.is_valid
                validation_message = validation.message

                if not validation.is_valid:
                    logger.info(
                        f"Image rejected: {validation.rejection_reason} "
                        f"(positive={validation.positive_score:.3f}, "
                        f"negative={validation.negative_score:.3f})"
                    )
                    _duration_ms = (time.time() - _start) * 1000
                    return {
                        **state,
                        "uploaded_image_validated": False,
                        "image_validation_message": validation.message,
                        "image_search_results": [],
                        "has_image_query": True,
                        "step_results": [{
                            "node": "vision_retrieval",
                            "status": "warning",
                            "summary": f"Image rejected: {validation.rejection_reason}",
                            "duration_ms": round(_duration_ms, 2),
                        }],
                    }

            except Exception as e:
                logger.warning(f"Image validation skipped (error: {e}), proceeding with search")
                validated = True
                validation_message = "Validation skipped"

            # Step 2: Image-to-Image search
            image_results = service.search_by_image_base64(
                uploaded_image, top_k=top_k, location_filter=target_location,
            )
            logger.info(f"Image-to-Image search returned {len(image_results)} results")

        else:
            # --- Text-to-Image flow ---
            image_results = service.search_by_text(
                query, top_k=top_k, location_filter=target_location,
            )
            logger.info(f"Text-to-Image search returned {len(image_results)} results")

    except Exception as e:
        logger.error(f"Vision retrieval failed: {e}")
        _duration_ms = (time.time() - _start) * 1000
        return {
            **state,
            "image_search_results": [],
            "has_image_query": True,
            "error": f"Vision retrieval failed: {e}",
            "step_results": [{
                "node": "vision_retrieval",
                "status": "error",
                "summary": f"Vision retrieval error: {str(e)[:100]}",
                "duration_ms": round(_duration_ms, 2),
            }],
        }

    # Build summary
    if image_results:
        locations_found = list(set(r["location_name"] for r in image_results))
        top_score = image_results[0]["similarity_score"] if image_results else 0
        summary = (
            f"Found {len(image_results)} images | "
            f"Top: {locations_found[0]} ({top_score:.3f}) | "
            f"Locations: {', '.join(locations_found[:3])}"
        )
    else:
        summary = "No matching images found"

    _duration_ms = (time.time() - _start) * 1000

    update: Dict = {
        **state,
        "image_search_results": image_results,
        "has_image_query": True,
        "step_results": [{
            "node": "vision_retrieval",
            "status": "success",
            "summary": summary,
            "duration_ms": round(_duration_ms, 2),
        }],
    }

    if validated is not None:
        update["uploaded_image_validated"] = validated
    if validation_message is not None:
        update["image_validation_message"] = validation_message

    return update
