"""
LangGraph Workflow: Agentic Tour Guide Reasoning Loop.

This module defines the complete LangGraph StateGraph that implements
the multi-step reasoning workflow for the Travion AI Tour Guide.

Graph Architecture:
    ┌─────────┐
    │  START  │
    └────┬────┘
         │
    ┌────▼────┐
    │ Router  │──────────────────────────┐
    └────┬────┘                          │
         │ (tourism_query/trip_plan)     │ (greeting/off_topic)
    ┌────▼────┐                          │
    │Retrieval│                          │
    └────┬────┘                          │
         │                               │
    ┌────▼────┐                          │
    │ Grader  │───────────┐              │
    └────┬────┘           │              │
         │ (sufficient)   │ (insufficient)
    ┌────▼──────────┐ ┌───▼───┐          │
    │Shadow Monitor │ │Web    │          │
    └────┬──────────┘ │Search │          │
         │            └───┬───┘          │
         │◄───────────────┘              │
         │                               │
    ┌────▼────┐◄─────────────────────────┘
    │Generator│
    └────┬────┘
         │
    ┌────▼────┐
    │Verifier │──────┐
    └────┬────┘      │ (needs correction)
         │           │
    ┌────▼────┐◄─────┘
    │   END   │
    └─────────┘

Research Pattern:
    ReAct (Reasoning + Acting) with Self-Correction Loop
    - Each node performs a specific reasoning step
    - Conditional edges enable dynamic routing
    - Verifier can loop back for corrections
"""

import logging
import uuid
from typing import Optional, Dict, Any, List

# Tracing imports
try:
    from ..utils.tracing import (
        trace_node,
        init_langsmith,
        create_run_config,
        get_tracing_callback,
        log_rag_metrics,
    )
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def trace_node(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

# LangChain imports
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Gemini LLM (Primary - Free tier available)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: langchain_google_genai import failed: {e}")
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None
except Exception as e:
    print(f"WARNING: langchain_google_genai import error: {e}")
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None

# OpenAI LLM (Fallback)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: langchain_openai import failed: {e}")
    OPENAI_AVAILABLE = False
    ChatOpenAI = None
except Exception as e:
    print(f"WARNING: langchain_openai import error: {e}")
    OPENAI_AVAILABLE = False
    ChatOpenAI = None

# Ollama LLM (Local fallback)
try:
    from langchain_community.chat_models import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ChatOllama = None

from .state import GraphState, create_initial_state, TourPlanContext, UserPreferences
from .nodes import (
    router_node,
    retrieval_node,
    grader_node,
    web_search_node,
    shadow_monitor_node,
    clarification_node,
    generator_node,
    verifier_node,
    tour_plan_generator_node,
    hotel_search_node,
    advanced_search_node,
    selection_handler_node,
    restaurant_selection_handler_node,
    vision_retrieval_node,
    route_by_intent,
    route_after_grading,
    route_after_verification,
    route_after_clarification,
    route_to_plan_generator,
    route_to_hotel_search,
    should_trigger_advanced_search,
)
from ..config import settings

logger = logging.getLogger(__name__)


class TravionAgentGraph:
    """
    Agentic Tour Guide workflow using LangGraph.

    This class encapsulates the complete reasoning loop, including:
    - LLM initialization (Gemini, OpenAI, or Ollama)
    - Graph construction
    - Execution interface

    Attributes:
        llm: LangChain chat model (Gemini, OpenAI, or Ollama)
        graph: Compiled LangGraph StateGraph
        memory: Checkpoint memory for conversation persistence

    Example:
        >>> agent = TravionAgentGraph()
        >>> result = await agent.invoke("Plan a trip to Jungle Beach next full moon")
        >>> print(result["final_response"])
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7
    ):
        """
        Initialize the Travion Agent Graph.

        Args:
            llm_provider: "gemini", "openai", or "ollama" (defaults to settings.LLM_PROVIDER)
            model_name: Model to use (e.g., "gemini-1.5-flash", "gpt-4o-mini", or "llama3.1:8b")
            temperature: LLM sampling temperature
        """
        self.llm = None
        self.graph = None
        self.memory = None

        # Determine LLM provider
        provider = llm_provider or settings.LLM_PROVIDER

        # Initialize LLM based on provider
        if provider == "gemini" and GEMINI_AVAILABLE:
            self._init_gemini(model_name, temperature)
        elif provider == "openai" and OPENAI_AVAILABLE:
            self._init_openai(model_name, temperature)
        elif provider == "ollama" and OLLAMA_AVAILABLE:
            self._init_ollama(model_name, temperature)
        else:
            logger.warning(f"LLM provider '{provider}' not available, trying fallbacks...")
            # Try Gemini first (free tier), then OpenAI, then Ollama
            if GEMINI_AVAILABLE and settings.GOOGLE_API_KEY:
                self._init_gemini(model_name, temperature)
            elif OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
                self._init_openai(model_name, temperature)
            elif OLLAMA_AVAILABLE:
                self._init_ollama(model_name, temperature)

        # Build graph
        if LANGGRAPH_AVAILABLE:
            if self.llm is None:
                logger.error("No LLM available! Please configure at least one LLM provider.")
                logger.error("Options: Set GOOGLE_API_KEY for Gemini, OPENAI_API_KEY for OpenAI, or ensure Ollama is running.")
                raise RuntimeError(
                    "LLM initialization failed. Please configure GOOGLE_API_KEY, OPENAI_API_KEY, "
                    "or ensure Ollama is running at the configured URL."
                )
            self._build_graph()
        else:
            logger.error("LangGraph not available. Install with: pip install langgraph")

    def _init_gemini(self, model_name: Optional[str], temperature: float):
        """Initialize Google Gemini LLM."""
        if GEMINI_AVAILABLE:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name or settings.GEMINI_MODEL,
                    temperature=temperature,
                    google_api_key=settings.GOOGLE_API_KEY,
                    convert_system_message_to_human=True  # Gemini compatibility
                )
                logger.info(f"Gemini LLM initialized: {model_name or settings.GEMINI_MODEL}")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")
                # Fallback to OpenAI
                self._init_openai(model_name, temperature)

    def _init_openai(self, model_name: Optional[str], temperature: float):
        """Initialize OpenAI LLM."""
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=model_name or settings.OPENAI_MODEL,
                    temperature=temperature,
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"OpenAI LLM initialized: {model_name or settings.OPENAI_MODEL}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI: {e}")
                # Fallback to Ollama
                self._init_ollama(model_name, temperature)

    def _init_ollama(self, model_name: Optional[str], temperature: float):
        """Initialize Ollama LLM."""
        if OLLAMA_AVAILABLE:
            try:
                self.llm = ChatOllama(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=model_name or settings.OLLAMA_MODEL,
                    temperature=temperature
                )
                logger.info(f"Ollama LLM initialized: {model_name or settings.OLLAMA_MODEL}")
            except Exception as e:
                logger.warning(f"Could not initialize Ollama: {e}")

    def _build_graph(self) -> None:
        """
        Construct the LangGraph StateGraph with all nodes and edges.

        This method defines the complete agentic workflow:
        1. Router → classifies intent and extracts entities
        2. Retrieval → fetches relevant documents (if needed)
        3. Grader → assesses document relevance
        4. Web Search → fallback for insufficient docs
        5. Shadow Monitor → constraint checking
        6. Generator → produces response (or Tour Plan Generator for plans)
        7. Verifier → validates and may loop for correction
        """
        # Create the graph with our state schema
        workflow = StateGraph(GraphState)

        # Add all nodes
        workflow.add_node("router", self._router_wrapper)
        workflow.add_node("retrieve", self._retrieval_wrapper)
        workflow.add_node("grader", self._grader_wrapper)
        workflow.add_node("web_search", self._web_search_wrapper)
        workflow.add_node("shadow_monitor", self._shadow_monitor_wrapper)
        workflow.add_node("clarification", self._clarification_wrapper)
        workflow.add_node("generate", self._generator_wrapper)
        workflow.add_node("tour_plan_generate", self._tour_plan_generator_wrapper)
        workflow.add_node("hotel_search", self._hotel_search_wrapper)
        workflow.add_node("advanced_search", self._advanced_search_wrapper)
        workflow.add_node("selection_handler", self._selection_handler_wrapper)
        workflow.add_node("restaurant_selection_handler", self._restaurant_selection_handler_wrapper)
        workflow.add_node("vision_retrieve", self._vision_retrieval_wrapper)
        workflow.add_node("verify", self._verifier_wrapper)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "generate": "generate",           # Greetings/off-topic → direct generation
                "retrieve": "retrieve",            # Tourism queries → retrieval
                "web_search": "web_search",        # Real-time info → web search
                "tour_plan": "retrieve",           # Tour plan → retrieval first
                "hotel_search": "hotel_search",    # Hotel/restaurant queries → web search
                "advanced_search": "advanced_search",  # Advanced multi-step search
                "vision_retrieve": "vision_retrieve",  # Image queries → CLIP search
            }
        )

        # Vision Retrieve → Generate (fast path, skip grader/shadow monitor)
        workflow.add_edge("vision_retrieve", "generate")

        # Retrieval → Grader
        workflow.add_edge("retrieve", "grader")

        # Conditional edges from grader
        workflow.add_conditional_edges(
            "grader",
            route_after_grading,
            {
                "web_search": "web_search",      # Insufficient → web search
                "shadow_monitor": "shadow_monitor"  # Sufficient → shadow monitor
            }
        )

        # Web search → Shadow Monitor
        workflow.add_edge("web_search", "shadow_monitor")

        # Shadow Monitor → Generate or Clarification (for tour plans)
        workflow.add_conditional_edges(
            "shadow_monitor",
            self._route_after_shadow_monitor,
            {
                "generate": "generate",
                "clarification": "clarification"
            }
        )

        # Clarification → Tour Plan Generate or END (if clarification needed)
        workflow.add_conditional_edges(
            "clarification",
            self._route_after_clarification,
            {
                "tour_plan_generate": "tour_plan_generate",
                END: END
            }
        )

        # Generate → Verify
        workflow.add_edge("generate", "verify")

        # Tour Plan Generate → conditional: restaurant HITL or verify
        workflow.add_conditional_edges(
            "tour_plan_generate",
            self._route_after_tour_plan_generate,
            {
                "restaurant_selection_handler": "restaurant_selection_handler",
                "verify": "verify",
            }
        )

        # Restaurant Selection Handler → loop back to tour_plan_generate
        # (second pass will have selected_restaurant_ids set, so it
        #  skips the HITL check and generates the full plan)
        workflow.add_edge("restaurant_selection_handler", "tour_plan_generate")

        # Hotel Search → Verify
        workflow.add_edge("hotel_search", "verify")

        # Advanced Search → END (pauses for HITL via interrupt_before)
        # When the graph resumes with a selected_search_candidate_id,
        # it continues into selection_handler.
        workflow.add_conditional_edges(
            "advanced_search",
            self._route_after_advanced_search,
            {
                "selection_handler": "selection_handler",
                END: END,
            }
        )

        # Selection Handler → Verify
        workflow.add_edge("selection_handler", "verify")

        # Conditional edges from verifier
        workflow.add_conditional_edges(
            "verify",
            route_after_verification,
            {
                "generate": "generate",  # Needs correction → regenerate
                END: END                 # Approved → end
            }
        )

        # Compile the graph with interrupt support for HITL
        # The advanced_search node sets pending_user_selection=True;
        # the graph then ends. When the caller resumes with
        # selected_search_candidate_id populated, it enters
        # selection_handler to re-optimise the itinerary.
        self.memory = MemorySaver()
        self.graph = workflow.compile(
            checkpointer=self.memory,
            interrupt_before=["selection_handler", "restaurant_selection_handler"],
        )

        logger.info("LangGraph workflow compiled successfully")

    def _route_after_router(self, state: GraphState) -> str:
        """Route after router based on intent and tour plan context."""
        # Check if this is a tour plan request
        if state.get("tour_plan_context"):
            return "tour_plan"

        # Check if user uploaded an image → always vision retrieval
        if state.get("uploaded_image_base64"):
            return "vision_retrieve"

        # Check if this is an advanced multi-step search
        if should_trigger_advanced_search(state.get("user_query", "")):
            return "advanced_search"

        # Check if this is a hotel/restaurant search (legacy fallback)
        if route_to_hotel_search(state):
            return "hotel_search"

        # Otherwise use standard intent routing
        return route_by_intent(state)

    def _route_after_advanced_search(self, state: GraphState) -> str:
        """Route after advanced search.

        If a user selection was already provided (graph resumed), proceed
        to selection_handler.  Otherwise, the graph ends so the mobile
        app can present the NEED_USER_SELECTION state.
        """
        if state.get("selected_search_candidate_id"):
            return "selection_handler"
        # End the graph — the interrupt_before on selection_handler will
        # cause the graph to pause until the user provides a selection.
        if state.get("pending_user_selection"):
            return "selection_handler"  # Will be intercepted by interrupt_before
        return END

    def _route_after_shadow_monitor(self, state: GraphState) -> str:
        """Route after shadow monitor.

        If a weather interrupt was triggered (rain > 80% or wind > 60 km/h),
        END the graph immediately so the backend can return a USER_PROMPT_REQUIRED
        state to the mobile app.  Otherwise continue to clarification (tour plans)
        or generate.
        """
        if state.get("weather_interrupt"):
            return END
        if state.get("tour_plan_context"):
            return "clarification"
        return "generate"

    def _route_after_clarification(self, state: GraphState) -> str:
        """Route after clarification: end if user input needed, else generate plan."""
        if state.get("clarification_needed"):
            # Short-circuit: return the question to the user
            # Set final_response so the graph returns properly
            question = state.get("clarification_question", {})
            return END
        return "tour_plan_generate"

    def _route_after_tour_plan_generate(self, state: GraphState) -> str:
        """Route after tour plan generation.

        If the node returned with ``pending_user_selection=True`` (restaurant
        HITL), route to restaurant_selection_handler which will be
        intercepted by ``interrupt_before``.  Otherwise proceed to verify.
        """
        if state.get("pending_restaurant_selection"):
            return "restaurant_selection_handler"
        return "verify"

    # Node wrappers that inject LLM
    async def _router_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for router node with LLM injection."""
        return await router_node(state, self.llm)

    async def _retrieval_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for retrieval node."""
        return await retrieval_node(state)

    async def _grader_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for grader node with LLM injection."""
        return await grader_node(state, self.llm)

    async def _web_search_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for web search node."""
        return await web_search_node(state)

    async def _shadow_monitor_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for shadow monitor node."""
        return await shadow_monitor_node(state)

    async def _clarification_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for clarification node."""
        return await clarification_node(state, self.llm)

    async def _generator_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for generator node with LLM injection."""
        return await generator_node(state, self.llm)

    async def _tour_plan_generator_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for tour plan generator node with LLM injection."""
        return await tour_plan_generator_node(state, self.llm)

    async def _hotel_search_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for hotel search node."""
        return await hotel_search_node(state, self.llm)

    async def _advanced_search_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for advanced multi-step search node."""
        return await advanced_search_node(state, self.llm)

    async def _selection_handler_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for HITL selection handler / itinerary re-optimization."""
        return await selection_handler_node(state, self.llm)

    async def _restaurant_selection_handler_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for restaurant HITL selection handler."""
        return await restaurant_selection_handler_node(state, self.llm)

    async def _vision_retrieval_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for vision retrieval node (CLIP image search)."""
        return await vision_retrieval_node(state)

    async def _verifier_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for verifier node with LLM injection."""
        return await verifier_node(state, self.llm)

    async def invoke(
        self,
        query: str,
        thread_id: Optional[str] = None,
        target_location: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        tour_plan_context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        uploaded_image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the agent workflow for a user query.

        Args:
            query: User's input message
            thread_id: Optional thread ID for conversation persistence
            target_location: Optional location name to focus retrieval on
            conversation_history: Optional list of previous messages for context
            tour_plan_context: Optional context for tour plan generation
            user_preferences: Optional user preference profile for personalization
            uploaded_image_base64: Optional base64-encoded image for CLIP visual search

        Returns:
            Dict with final state including response, step results, and cultural tips
        """
        if not self.graph:
            return {
                "error": "Graph not initialized",
                "final_response": "I'm having trouble processing your request. Please try again."
            }

        # Create initial state with all optional parameters
        initial_state = create_initial_state(
            query,
            target_location=target_location,
            conversation_history=conversation_history,
            tour_plan_context=tour_plan_context,
            user_preferences=user_preferences,
            uploaded_image_base64=uploaded_image_base64,
        )

        # Configure thread with tracing metadata
        if TRACING_AVAILABLE:
            config = create_run_config(
                thread_id=thread_id,
                user_query=query,
                target_location=target_location,
                tags=["chat", target_location or "general"],
                metadata={
                    "has_tour_plan_context": bool(tour_plan_context),
                    "has_user_preferences": bool(user_preferences),
                    "conversation_length": len(conversation_history) if conversation_history else 0,
                }
            )
        else:
            config = {
                "configurable": {
                    "thread_id": thread_id or str(uuid.uuid4())
                }
            }

        try:
            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state, config)

            # Log RAG metrics for monitoring
            if TRACING_AVAILABLE:
                log_rag_metrics(
                    query=query,
                    documents_retrieved=len(final_state.get("retrieved_documents", [])),
                    relevance_score=final_state.get("document_relevance", {}).get("score", 0.0) if isinstance(final_state.get("document_relevance"), dict) else 0.0,
                    response_length=len(final_state.get("final_response", "") or ""),
                    reasoning_loops=final_state.get("reasoning_loops", 0)
                )

            return {
                "query": query,
                "intent": final_state.get("intent").value if final_state.get("intent") else None,
                "final_response": final_state.get("final_response") or final_state.get("generated_response"),
                "itinerary": final_state.get("itinerary"),
                "tour_plan_metadata": final_state.get("tour_plan_metadata"),
                "constraint_violations": final_state.get("constraint_violations"),
                "shadow_monitor_logs": final_state.get("shadow_monitor_logs"),
                "reasoning_loops": final_state.get("reasoning_loops", 0),
                "documents_retrieved": len(final_state.get("retrieved_documents", [])),
                "web_search_used": len(final_state.get("web_search_results", [])) > 0,
                # New fields for super-accuracy
                "step_results": final_state.get("step_results", []),
                "clarification_needed": final_state.get("clarification_needed", False),
                "clarification_question": final_state.get("clarification_question"),
                "cultural_tips": final_state.get("cultural_tips", []),
                # New fields for super-optimized agent
                "final_itinerary": final_state.get("final_itinerary"),
                "weather_data": final_state.get("weather_data"),
                "hotel_search_results": final_state.get("hotel_search_results", []),
                "interrupt_reason": final_state.get("interrupt_reason"),
                # Advanced Multi-Step Search & HITL fields
                "search_candidates": final_state.get("search_candidates", []),
                "pending_user_selection": final_state.get("pending_user_selection", False),
                "selected_search_candidate": final_state.get("selected_search_candidate"),
                # MCP Search — Selection Cards & Map-Ready Itinerary
                "selection_cards": final_state.get("selection_cards"),
                "prompt_text": final_state.get("prompt_text"),
                "mcp_search_metadata": final_state.get("mcp_search_metadata"),
                "map_ready_itinerary": final_state.get("map_ready_itinerary"),
                # Weather Interrupt — USER_PROMPT_REQUIRED
                "weather_interrupt": final_state.get("weather_interrupt", False),
                "weather_prompt_message": final_state.get("weather_prompt_message"),
                "weather_prompt_options": final_state.get("weather_prompt_options"),
                # Vision / Image Search results
                "image_search_results": final_state.get("image_search_results", []),
                "has_image_query": final_state.get("has_image_query", False),
                "uploaded_image_validated": final_state.get("uploaded_image_validated"),
                "image_validation_message": final_state.get("image_validation_message"),
            }

        except Exception as e:
            import traceback
            logger.error(f"Graph execution failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "query": query,
                "final_response": "I encountered an error while processing your request. Please try again."
            }

    async def resume_with_selection(
        self,
        thread_id: str,
        selected_candidate_id: str,
    ) -> Dict[str, Any]:
        """
        Resume a paused graph after the user selects a search candidate.

        This is the second half of the Human-in-the-Loop (HITL) flow:
        1. The graph paused at the interrupt_before on selection_handler.
        2. The mobile app presented candidates to the user.
        3. The user selected a candidate, calling this method.
        4. The graph resumes: selection_handler validates the pick,
           recalculates Haversine travel times, and re-checks Event Sentinel.

        Args:
            thread_id: The same thread_id from the original invoke.
            selected_candidate_id: The `id` of the chosen SearchCandidate.

        Returns:
            Dict with the final state after re-optimization.
        """
        if not self.graph:
            return {
                "error": "Graph not initialized",
                "final_response": "Unable to process selection — graph not ready.",
            }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            # LangGraph 1.0.x resume pattern for interrupt_before:
            # 1. Update the paused checkpoint state with the user's selection
            # 2. Resume by calling ainvoke(None) — NOT ainvoke(update)
            update = {"selected_search_candidate_id": selected_candidate_id}
            await self.graph.aupdate_state(config, update)
            final_state = await self.graph.ainvoke(None, config)

            return {
                "query": final_state.get("user_query", ""),
                "intent": final_state.get("intent").value if final_state.get("intent") else None,
                "final_response": final_state.get("final_response") or final_state.get("generated_response"),
                "itinerary": final_state.get("itinerary"),
                "tour_plan_metadata": final_state.get("tour_plan_metadata"),
                "final_itinerary": final_state.get("final_itinerary"),
                "constraint_violations": final_state.get("constraint_violations", []),
                "step_results": final_state.get("step_results", []),
                "cultural_tips": final_state.get("cultural_tips", []),
                "weather_data": final_state.get("weather_data"),
                "restaurant_recommendations": final_state.get("restaurant_recommendations", []),
                "accommodation_recommendations": final_state.get("accommodation_recommendations", []),
                "search_candidates": final_state.get("search_candidates", []),
                "selected_search_candidate": final_state.get("selected_search_candidate"),
                "pending_user_selection": final_state.get("pending_user_selection", False),
                # MCP — Map-Ready Itinerary & Selection Cards
                "selection_cards": final_state.get("selection_cards"),
                "prompt_text": final_state.get("prompt_text"),
                "mcp_search_metadata": final_state.get("mcp_search_metadata"),
                "map_ready_itinerary": final_state.get("map_ready_itinerary"),
                # Weather interrupt fields (in case another interrupt fires)
                "weather_interrupt": final_state.get("weather_interrupt", False),
                "weather_prompt_message": final_state.get("weather_prompt_message"),
                "weather_prompt_options": final_state.get("weather_prompt_options"),
            }

        except Exception as e:
            import traceback
            logger.error(f"Resume with selection failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "final_response": "I encountered an error processing your selection. Please try again.",
            }

    async def resume_with_weather_choice(
        self,
        thread_id: str,
        user_choice: str,
    ) -> Dict[str, Any]:
        """
        Resume a graph paused due to a USER_PROMPT_REQUIRED weather interrupt.

        The mobile app showed the weather prompt and the user picked one of:
            - ``switch_indoor``: Replace with an indoor alternative
            - ``reschedule``: Same activity, different time
            - ``keep``: Proceed despite bad weather

        The graph resumes from the shadow_monitor checkpoint, feeds the
        choice back, and lets the downstream nodes (generator / tour
        plan generator) incorporate it.

        Args:
            thread_id: Thread ID from the original invoke.
            user_choice: One of ``switch_indoor``, ``reschedule``, ``keep``.

        Returns:
            Dict with the final state after re-processing.
        """
        if not self.graph:
            return {
                "error": "Graph not initialized",
                "final_response": "Unable to process weather choice — graph not ready.",
            }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            # LangGraph 1.0.x resume pattern: update_state then ainvoke(None)
            update = {
                "user_weather_choice": user_choice,
                "weather_interrupt": False,       # clear the interrupt flag
                "interrupt_reason": None,
            }
            await self.graph.aupdate_state(config, update)
            final_state = await self.graph.ainvoke(None, config)

            return {
                "query": final_state.get("user_query", ""),
                "intent": final_state.get("intent").value if final_state.get("intent") else None,
                "final_response": final_state.get("final_response") or final_state.get("generated_response"),
                "itinerary": final_state.get("itinerary"),
                "final_itinerary": final_state.get("final_itinerary"),
                "constraint_violations": final_state.get("constraint_violations", []),
                "step_results": final_state.get("step_results", []),
                "weather_interrupt": False,
                "map_ready_itinerary": final_state.get("map_ready_itinerary"),
            }

        except Exception as e:
            import traceback
            logger.error(f"Resume with weather choice failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {
                "error": str(e),
                "final_response": "I encountered an error processing your weather choice. Please try again.",
            }

    async def stream(
        self,
        query: str,
        thread_id: Optional[str] = None,
        target_location: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Stream the agent execution step by step.

        This method yields intermediate states as the graph executes,
        enabling real-time visibility into the reasoning process.

        Args:
            query: User's input message
            thread_id: Optional thread ID
            target_location: Optional location name to focus retrieval on
            conversation_history: Optional list of previous messages for context

        Yields:
            Dict with node name and current state
        """
        if not self.graph:
            yield {"error": "Graph not initialized"}
            return

        initial_state = create_initial_state(
            query, 
            target_location=target_location,
            conversation_history=conversation_history
        )
        config = {
            "configurable": {
                "thread_id": thread_id or str(uuid.uuid4())
            }
        }

        try:
            async for event in self.graph.astream(initial_state, config):
                for node_name, node_state in event.items():
                    yield {
                        "node": node_name,
                        "state": {
                            k: v for k, v in node_state.items()
                            if k not in ["messages"]  # Exclude large fields
                        }
                    }

        except Exception as e:
            logger.error(f"Stream failed: {e}")
            yield {"error": str(e)}

    def get_graph_visualization(self) -> str:
        """
        Get a Mermaid diagram representation of the graph.

        Returns:
            str: Mermaid diagram code

        Example:
            >>> print(agent.get_graph_visualization())
            # Copy to mermaid.live to visualize
        """
        if not self.graph:
            return "Graph not initialized"

        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            # Fallback to manual diagram
            return """
graph TD
    START([Start]) --> router[Router]
    router -->|greeting/off_topic| generate[Generator]
    router -->|tourism_query| retrieve[Retrieval]
    router -->|real_time_info| web_search[Web Search]
    retrieve --> grader[Grader]
    grader -->|sufficient| shadow_monitor[Shadow Monitor]
    grader -->|insufficient| web_search
    web_search --> shadow_monitor
    shadow_monitor --> generate
    shadow_monitor -->|tour_plan| tour_plan_generate[Tour Plan Generator]
    tour_plan_generate --> verify
    generate --> verify[Verifier]
    verify -->|approved| END([End])
    verify -->|needs_correction| generate
"""


# Singleton agent instance
_agent: Optional[TravionAgentGraph] = None


def get_agent() -> TravionAgentGraph:
    """
    Get or create the TravionAgentGraph singleton.

    Returns:
        TravionAgentGraph: Singleton agent instance
    """
    global _agent
    if _agent is None:
        _agent = TravionAgentGraph()
    return _agent


async def invoke_agent(
    query: str,
    thread_id: Optional[str] = None,
    target_location: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    tour_plan_context: Optional[Dict[str, Any]] = None,
    user_preferences: Optional[Dict[str, Any]] = None,
    uploaded_image_base64: Optional[str] = None,
) -> Dict:
    """
    Convenience function to invoke the agent.

    Args:
        query: User's query
        thread_id: Optional conversation thread ID
        target_location: Optional location name to focus retrieval on
        conversation_history: Optional list of previous messages for context
        tour_plan_context: Optional context for tour plan generation
        user_preferences: Optional user preference profile for personalization
        uploaded_image_base64: Optional base64-encoded image for CLIP visual search

    Returns:
        Dict with agent response including step_results, cultural_tips, clarification
    """
    agent = get_agent()
    return await agent.invoke(
        query,
        thread_id,
        target_location=target_location,
        conversation_history=conversation_history,
        tour_plan_context=tour_plan_context,
        user_preferences=user_preferences,
        uploaded_image_base64=uploaded_image_base64,
    )


async def resume_agent_with_selection(
    thread_id: str,
    selected_candidate_id: str,
) -> Dict:
    """
    Convenience function to resume the agent after a HITL selection.

    Args:
        thread_id: Thread ID from the original invoke.
        selected_candidate_id: The ID of the candidate the user selected.

    Returns:
        Dict with re-optimized itinerary and selection result.
    """
    agent = get_agent()
    return await agent.resume_with_selection(thread_id, selected_candidate_id)


async def resume_agent_with_weather_choice(
    thread_id: str,
    user_choice: str,
) -> Dict:
    """
    Convenience function to resume the agent after a weather interrupt.

    Args:
        thread_id: Thread ID from the original invoke.
        user_choice: One of "switch_indoor", "reschedule", or "keep".

    Returns:
        Dict with the resumed graph result.
    """
    agent = get_agent()
    return await agent.resume_with_weather_choice(thread_id, user_choice)
