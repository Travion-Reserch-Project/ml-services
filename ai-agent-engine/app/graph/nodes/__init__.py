"""
LangGraph Nodes Package for Travion AI Engine.

This package contains all the nodes that form the agentic reasoning loop:

Nodes:
    - router: Intent classification and entity extraction
    - retrieval: ChromaDB vector search
    - grader: Document relevance assessment (self-correction)
    - web_search: Tavily fallback for real-time info
    - shadow_monitor: Multi-constraint validation
    - clarification: Interactive agent questioning (human-in-the-loop)
    - generator: LLM response generation
    - verifier: Final self-correction check
    - tour_plan_generator: Multi-day itinerary generation with deep data injection

Research Architecture:
    The nodes implement a "Reflective RAG" pattern with:
    1. Perception (router, retrieval)
    2. Reflection (grader, shadow_monitor)
    3. Clarification (clarification — human-in-the-loop questioning)
    4. Action (generator, tour_plan_generator)
    5. Verification (verifier with correction loop)
"""

from .router import router_node, route_by_intent, extract_entities
from .retrieval import retrieval_node, get_vectordb_service
from .grader import grader_node, route_after_grading
from .web_search import web_search_node
from .shadow_monitor import shadow_monitor_node, get_shadow_monitor
from .clarification import clarification_node, route_after_clarification
from .generator import generator_node
from .verifier import verifier_node, route_after_verification
from .tour_plan_generator import tour_plan_generator_node, route_to_plan_generator
from .hotel_search import hotel_search_node, route_to_hotel_search
from .advanced_search import (
    advanced_search_node,
    should_trigger_advanced_search,
    detect_search_type,
)
from .selection_handler import selection_handler_node
from .restaurant_selection_handler import restaurant_selection_handler_node

__all__ = [
    # Nodes
    "router_node",
    "retrieval_node",
    "grader_node",
    "web_search_node",
    "shadow_monitor_node",
    "clarification_node",
    "generator_node",
    "verifier_node",
    "tour_plan_generator_node",
    "hotel_search_node",
    "advanced_search_node",
    "selection_handler_node",
    "restaurant_selection_handler_node",
    # Routing functions
    "route_by_intent",
    "route_after_grading",
    "route_after_verification",
    "route_after_clarification",
    "route_to_plan_generator",
    "route_to_hotel_search",
    "should_trigger_advanced_search",
    "detect_search_type",
    # Utilities
    "extract_entities",
    "get_vectordb_service",
    "get_shadow_monitor",
]
