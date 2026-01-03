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

from .state import GraphState, create_initial_state, TourPlanContext
from .nodes import (
    router_node,
    retrieval_node,
    grader_node,
    web_search_node,
    shadow_monitor_node,
    generator_node,
    verifier_node,
    tour_plan_generator_node,
    route_by_intent,
    route_after_grading,
    route_after_verification,
    route_to_plan_generator,
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
        workflow.add_node("generate", self._generator_wrapper)
        workflow.add_node("tour_plan_generate", self._tour_plan_generator_wrapper)
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
                "tour_plan": "retrieve"            # Tour plan → retrieval first
            }
        )

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

        # Shadow Monitor → Generate or Tour Plan Generate
        workflow.add_conditional_edges(
            "shadow_monitor",
            self._route_after_shadow_monitor,
            {
                "generate": "generate",
                "tour_plan_generate": "tour_plan_generate"
            }
        )

        # Generate → Verify
        workflow.add_edge("generate", "verify")
        
        # Tour Plan Generate → Verify
        workflow.add_edge("tour_plan_generate", "verify")

        # Conditional edges from verifier
        workflow.add_conditional_edges(
            "verify",
            route_after_verification,
            {
                "generate": "generate",  # Needs correction → regenerate
                END: END                 # Approved → end
            }
        )

        # Compile the graph
        self.memory = MemorySaver()
        self.graph = workflow.compile(checkpointer=self.memory)

        logger.info("LangGraph workflow compiled successfully")

    def _route_after_router(self, state: GraphState) -> str:
        """Route after router based on intent and tour plan context."""
        # Check if this is a tour plan request
        if state.get("tour_plan_context"):
            return "tour_plan"
        
        # Otherwise use standard intent routing
        return route_by_intent(state)

    def _route_after_shadow_monitor(self, state: GraphState) -> str:
        """Route after shadow monitor to either generator or tour plan generator."""
        if state.get("tour_plan_context"):
            return "tour_plan_generate"
        return "generate"

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

    async def _generator_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for generator node with LLM injection."""
        return await generator_node(state, self.llm)

    async def _tour_plan_generator_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for tour plan generator node with LLM injection."""
        return await tour_plan_generator_node(state, self.llm)

    async def _verifier_wrapper(self, state: GraphState) -> GraphState:
        """Wrapper for verifier node with LLM injection."""
        return await verifier_node(state, self.llm)

    async def invoke(
        self,
        query: str,
        thread_id: Optional[str] = None,
        target_location: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        tour_plan_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the agent workflow for a user query.

        Args:
            query: User's input message
            thread_id: Optional thread ID for conversation persistence
            target_location: Optional location name to focus retrieval on
            conversation_history: Optional list of previous messages for context
            tour_plan_context: Optional context for tour plan generation

        Returns:
            Dict with final state including response and logs

        Example:
            >>> result = await agent.invoke("What's special about Sigiriya?")
            >>> print(result["final_response"])

            >>> # Location-specific chat
            >>> result = await agent.invoke("What's the best time to visit?", target_location="Sigiriya")

            >>> # With conversation history
            >>> history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
            >>> result = await agent.invoke("Tell me more", conversation_history=history)
            
            >>> # Tour plan generation
            >>> context = {"selected_locations": [...], "start_date": "2026-01-05", "end_date": "2026-01-07"}
            >>> result = await agent.invoke("Generate my tour plan", tour_plan_context=context)
        """
        if not self.graph:
            return {
                "error": "Graph not initialized",
                "final_response": "I'm having trouble processing your request. Please try again."
            }

        # Create initial state with optional target location, conversation history, and tour plan context
        initial_state = create_initial_state(
            query, 
            target_location=target_location,
            conversation_history=conversation_history,
            tour_plan_context=tour_plan_context
        )

        # Configure thread (always required when using checkpointer)
        config = {
            "configurable": {
                "thread_id": thread_id or str(uuid.uuid4())
            }
        }

        try:
            # Execute the graph
            final_state = await self.graph.ainvoke(initial_state, config)

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
                "web_search_used": len(final_state.get("web_search_results", [])) > 0
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
    tour_plan_context: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Convenience function to invoke the agent.

    Args:
        query: User's query
        thread_id: Optional conversation thread ID
        target_location: Optional location name to focus retrieval on
        conversation_history: Optional list of previous messages for context
        tour_plan_context: Optional context for tour plan generation

    Returns:
        Dict with agent response
    """
    agent = get_agent()
    return await agent.invoke(
        query, 
        thread_id, 
        target_location=target_location, 
        conversation_history=conversation_history,
        tour_plan_context=tour_plan_context
    )
