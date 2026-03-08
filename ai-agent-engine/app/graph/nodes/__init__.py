"""
LangGraph Nodes Package for Travion AI Engine.

This package contains all the nodes that form the agentic reasoning loop:

Nodes:
    - router: Intent classification and entity extraction
    - retrieval: ChromaDB vector search
    - grader: Document relevance assessment (self-correction)
    - web_search: Tavily fallback for real-time info
    - shadow_monitor: Multi-constraint validation
    - generator: LLM response generation
    - verifier: Final self-correction check
    - tour_plan_generator: Multi-day itinerary generation

Research Architecture:
    The nodes implement a "Reflective RAG" pattern with:
    1. Perception (router, retrieval)
    2. Reflection (grader, shadow_monitor)
    3. Action (generator, tour_plan_generator)
    4. Verification (verifier with correction loop)
"""

from .router import router_node, route_by_intent, extract_entities
from .retrieval import retrieval_node, get_vectordb_service
from .grader import grader_node, route_after_grading
from .web_search import web_search_node
from .shadow_monitor import shadow_monitor_node, get_shadow_monitor
from .generator import generator_node
from .verifier import verifier_node, route_after_verification
from .tour_plan_generator import tour_plan_generator_node, route_to_plan_generator

__all__ = [
    # Nodes
    "router_node",
    "retrieval_node",
    "grader_node",
    "web_search_node",
    "shadow_monitor_node",
    "generator_node",
    "verifier_node",
    "tour_plan_generator_node",
    # Routing functions
    "route_by_intent",
    "route_after_grading",
    "route_after_verification",
    "route_to_plan_generator",
    # Utilities
    "extract_entities",
    "get_vectordb_service",
    "get_shadow_monitor",
]
