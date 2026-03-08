"""
RAG Service

Orchestrates Retrieval-Augmented Generation for transport chatbot.
Combines vector search, text processing, and LLM generation.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.schemas.knowledge import (
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    SearchResult
)
from app.services.vector_db_service import get_vector_db_service
from app.utils.straico_client import StraicoClient, truncate_context
from app.utils.text_processor import get_text_processor

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG orchestration service
    
    Workflow:
    1. Preprocess user query
    2. Detect language
    3. Search vector database
    4. Build context from results
    5. Generate answer using Straico LLM
    6. Format and return response
    """
    
    def __init__(self):
        """Initialize RAG service components"""
        
        # Core components
        self.vector_db = get_vector_db_service()
        self.text_processor = get_text_processor()
        self.straico_client = StraicoClient()
        
        # Configuration
        self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))
        self.default_top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.min_similarity = float(os.getenv("RAG_MIN_SIMILARITY", "0.7"))
        
        # Load prompt templates
        self.prompts = self._load_prompts()
        
        logger.info("✓ RAG service initialized")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files"""
        prompts = {}
        prompt_dir = Path("data/prompts")
        
        try:
            # Load search prompt
            search_prompt_path = prompt_dir / "search_prompt.txt"
            if search_prompt_path.exists():
                with open(search_prompt_path, 'r', encoding='utf-8') as f:
                    prompts['search'] = f.read()
            else:
                prompts['search'] = self._get_default_search_prompt()
            
            # Load fare query prompt
            fare_prompt_path = prompt_dir / "fare_query_prompt.txt"
            if fare_prompt_path.exists():
                with open(fare_prompt_path, 'r', encoding='utf-8') as f:
                    prompts['fare'] = f.read()
            else:
                prompts['fare'] = prompts['search']
            
            logger.info(f"✓ Loaded {len(prompts)} prompt templates")
            
        except Exception as e:
            logger.warning(f"Failed to load prompts: {e}, using defaults")
            prompts['search'] = self._get_default_search_prompt()
            prompts['fare'] = prompts['search']
        
        return prompts
    
    def _get_default_search_prompt(self) -> str:
        """Get default search prompt"""
        return """You are a helpful Sri Lankan transport assistant.

Context Information:
{context}

User Question: {query}

Instructions:
- Provide accurate fare information from the context
- Format currency as LKR (Sri Lankan Rupees)
- Be conversational and helpful
- If you don't have enough information, say so clearly

Answer:"""
    
    def search_knowledge(
        self,
        request: KnowledgeSearchRequest
    ) -> KnowledgeSearchResponse:
        """
        Main RAG pipeline: search + generate
        
        Args:
            request: Search request with query and parameters
            
        Returns:
            Search response with results and optional generated answer
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess query
            processed_query = self.text_processor.preprocess_query(request.query)
            
            # Step 2: Detect language
            if request.language == "auto":
                detected_lang = self.text_processor.detect_language(request.query)
            else:
                detected_lang = request.language
            
            logger.info(f"Query language: {detected_lang}")
            
            # Step 3: Vector search
            search_start = time.time()
            
            collection = request.collection_name or "transport_knowledge"
            results = self.vector_db.search(
                query=processed_query,
                collection_name=collection,
                top_k=request.top_k,
                filters=request.filters
            )
            
            search_time = (time.time() - search_start) * 1000  # ms
            
            # Filter by minimum similarity only when score is on a 0..1 scale.
            # Some Chroma distance spaces can yield values that make (1 - distance)
            # negative, so strict similarity filtering would incorrectly drop all hits.
            bounded_scores = [r for r in results if 0.0 <= r.score <= 1.0]
            if bounded_scores:
                results = [r for r in bounded_scores if r.score >= self.min_similarity]
            elif results:
                logger.warning(
                    "Search scores are outside 0..1 range; skipping min_similarity filter"
                )
            
            logger.info(f"Found {len(results)} relevant documents (search: {search_time:.0f}ms)")
            
            # Step 4: Generate answer (if requested)
            answer = None
            generated_by = None
            rag_time = None
            
            if request.use_rag and results:
                rag_start = time.time()
                
                answer, generated_by = self._generate_answer(
                    query=request.query,
                    results=results,
                    language=detected_lang
                )
                
                rag_time = (time.time() - rag_start) * 1000  # ms
                logger.info(f"Generated answer (RAG: {rag_time:.0f}ms)")
            
            # Step 5: Build response
            total_time = (time.time() - start_time) * 1000  # ms
            
            return KnowledgeSearchResponse(
                success=True,
                query=request.query,
                results=results,
                answer=answer,
                generated_by=generated_by,
                query_language=detected_lang,
                total_results=len(results),
                search_time_ms=search_time,
                rag_time_ms=rag_time
            )
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}", exc_info=True)
            
            return KnowledgeSearchResponse(
                success=False,
                query=request.query,
                results=[],
                total_results=0
            )
    
    def _generate_answer(
        self,
        query: str,
        results: List[SearchResult],
        language: str
    ) -> tuple[str, str]:
        """
        Generate answer using LLM
        
        Args:
            query: User query
            results: Search results
            language: Detected language
            
        Returns:
            Tuple of (answer, model_name)
        """
        try:
            # Build context from results
            context = self._build_context(results)
            
            # Truncate if needed
            context = truncate_context(context, self.max_context_tokens)
            
            # Select prompt template
            if self.text_processor.is_fare_query(query):
                prompt_template = self.prompts.get('fare', self.prompts['search'])
            else:
                prompt_template = self.prompts['search']
            
            # Format prompt
            formatted_prompt = prompt_template.format(
                context=context,
                query=query
            )
            
            # Generate with Straico
            response = self.straico_client.generate(
                query=formatted_prompt,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.content, response.model
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            
            # Fallback: return structured results
            fallback = self._create_fallback_answer(results)
            return fallback, "fallback"
    
    def _build_context(self, results: List[SearchResult]) -> str:
        """
        Build context string from search results
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            # Format metadata
            meta = result.metadata
            
            meta_parts = []
            if meta.transport_type:
                meta_parts.append(f"Transport Type: {meta.transport_type}")
            if meta.category:
                meta_parts.append(f"Category: {meta.category}")
            if meta.route_name:
                meta_parts.append(f"Route: {meta.route_name}")
            if meta.route_number:
                meta_parts.append(f"Route Number: {meta.route_number}")
            if meta.location or meta.location_en:
                meta_parts.append(f"Location: {meta.location or meta.location_en}")
            if meta.source:
                meta_parts.append(f"Source: {meta.source}")
            
            # Build context entry
            context_entry = f"[Document {i}]"
            if meta_parts:
                context_entry += f"\n{' | '.join(meta_parts)}"
            context_entry += f"\n{result.content}\n"
            
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _create_fallback_answer(self, results: List[SearchResult]) -> str:
        """
        Create fallback answer from results (when LLM fails)
        
        Args:
            results: Search results
            
        Returns:
            Structured fallback answer
        """
        if not results:
            return "I couldn't find any relevant information for your query."
        
        answer = "Here is what I found from your transport data:\n\n"

        for i, result in enumerate(results[:3], 1):  # Top 3 results
            meta = result.metadata
            summary = meta.route_name or meta.location or meta.location_en or meta.category or "General transport info"

            # Try to parse JSON-style city records for a more human fallback summary.
            parsed_city_line = None
            content_str = result.content.strip()
            if content_str.startswith("{") and "city_name" in content_str:
                try:
                    import json
                    obj = json.loads(content_str)
                    city = obj.get("city_name")
                    rail = obj.get("has_railway_access")
                    station = (obj.get("nearest_railway_station") or {}).get("station_name")
                    dist = obj.get("distance_to_nearest_railway_km")
                    if city is not None and rail is not None:
                        rail_text = "has" if rail else "does not have"
                        if station:
                            parsed_city_line = (
                                f"{city} {rail_text} railway access; nearest station is {station}"
                                + (f" ({dist} km)." if dist is not None else ".")
                            )
                        else:
                            parsed_city_line = f"{city} {rail_text} railway access."
                except Exception:
                    parsed_city_line = None

            answer += f"{i}. {summary}\n"
            if parsed_city_line:
                answer += f"   {parsed_city_line}\n"
            if meta.transport_type:
                answer += f"   Type: {meta.transport_type}\n"
            if meta.source:
                answer += f"   Source: {meta.source}\n"
            if not parsed_city_line:
                answer += f"   Excerpt: {result.content[:180].replace(chr(10), ' ')}\n"
            answer += "\n"

        answer += "Tip: Ask a route-specific question like 'Best way from Colombo Fort to Kandy Station?' for a more targeted answer."
        return answer.strip()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of RAG components
        
        Returns:
            Health status dictionary
        """
        status = {
            "rag_service": "healthy",
            "components": {}
        }
        
        try:
            # Check vector DB
            collections = self.vector_db.list_collections()
            status["components"]["vector_db"] = {
                "status": "healthy",
                "collections": len(collections)
            }
        except Exception as e:
            status["components"]["vector_db"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        try:
            # Check Straico client
            status["components"]["llm"] = {
                "status": "healthy",
                "model": self.straico_client.model
            }
        except Exception as e:
            status["components"]["llm"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall status
        unhealthy = any(
            c.get("status") == "unhealthy" 
            for c in status["components"].values()
        )
        
        if unhealthy:
            status["rag_service"] = "degraded"
        
        return status


# ================================================
# Singleton Instance
# ================================================

_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create singleton RAG service instance"""
    global _rag_service
    
    if _rag_service is None:
        _rag_service = RAGService()
    
    return _rag_service


# ================================================
# Testing
# ================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test RAG service
    service = RAGService()
    
    # Test query
    from app.schemas.knowledge import KnowledgeSearchRequest
    
    request = KnowledgeSearchRequest(
        query="What is the bus fare from Colombo to Galle?",
        top_k=5,
        use_rag=True
    )
    
    response = service.search_knowledge(request)
    
    print(f"\n{'='*60}")
    print("RAG Query Results")
    print(f"{'='*60}")
    print(f"Query: {response.query}")
    print(f"Language: {response.query_language}")
    print(f"Results: {response.total_results}")
    print(f"Search time: {response.search_time_ms:.0f}ms")
    
    if response.answer:
        print(f"\n{'-'*60}")
        print("Generated Answer:")
        print(f"{'-'*60}")
        print(response.answer)
        print(f"\nGenerated by: {response.generated_by}")
        print(f"RAG time: {response.rag_time_ms:.0f}ms")
