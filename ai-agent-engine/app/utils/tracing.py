"""
LangSmith Tracing and Monitoring Utilities.

This module provides comprehensive tracing and monitoring capabilities
for the Travion RAG agentic system using LangSmith.

Features:
    - Node-level tracing for all graph nodes
    - Tool execution tracing
    - LLM call monitoring with latency metrics
    - Error tracking and alerting
    - Custom run metadata for filtering in LangSmith UI

LangSmith Dashboard:
    Once configured, you can view traces at: https://smith.langchain.com
    - View complete conversation flows
    - Debug retrieval quality
    - Monitor LLM latency and costs
    - Track error rates and patterns

Setup:
    1. Create account at https://smith.langchain.com
    2. Get API key from Settings
    3. Set LANGCHAIN_API_KEY in .env file
    4. Set LANGCHAIN_TRACING_V2=true in .env

Example:
    >>> from app.utils.tracing import trace_node, init_langsmith
    >>> init_langsmith()
    >>> 
    >>> @trace_node("my_node")
    >>> async def my_node_function(state):
    ...     return state
"""

import os
import logging
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])

# Global tracing state
_tracing_enabled = False
_langsmith_client = None
_project_name = "travion-ai-engine"

# Try to import LangSmith
try:
    from langsmith import Client, traceable
    from langsmith.run_trees import RunTree
    from langchain_core.tracers import LangChainTracer
    from langchain_core.callbacks import CallbackManager
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    traceable = None
    RunTree = None
    LangChainTracer = None
    CallbackManager = None
    logger.warning("LangSmith not available. Install with: pip install langsmith")


@dataclass
class TracingMetrics:
    """
    Metrics collected during node execution.
    
    Attributes:
        node_name: Name of the node
        start_time: Execution start timestamp
        end_time: Execution end timestamp
        latency_ms: Execution time in milliseconds
        success: Whether execution succeeded
        error_message: Error message if failed
        input_tokens: Approximate input token count
        output_tokens: Approximate output token count
        metadata: Additional metadata
    """
    node_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """Mark execution as complete and calculate latency."""
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_message = error_message
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "node_name": self.node_name,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms else None,
            "success": self.success,
            "error_message": self.error_message,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }


def init_langsmith(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None
) -> bool:
    """
    Initialize LangSmith tracing.
    
    This function sets up the environment variables and creates a LangSmith
    client for tracing. Call this at application startup.
    
    Args:
        api_key: LangSmith API key (defaults to LANGCHAIN_API_KEY env var)
        project_name: Project name for tracing (defaults to "travion-ai-engine")
        endpoint: LangSmith API endpoint (defaults to standard endpoint)
        
    Returns:
        bool: True if initialization succeeded, False otherwise
        
    Example:
        >>> from app.utils.tracing import init_langsmith
        >>> success = init_langsmith()
        >>> print(f"LangSmith enabled: {success}")
    """
    global _tracing_enabled, _langsmith_client, _project_name
    
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not available - tracing disabled")
        return False
    
    # Get configuration from environment or parameters
    api_key = api_key or os.getenv("LANGCHAIN_API_KEY")
    project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "travion-ai-engine")
    endpoint = endpoint or os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    
    if not api_key:
        logger.warning("No LANGCHAIN_API_KEY found - tracing disabled")
        return False
    
    try:
        # Set environment variables for LangChain automatic tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        
        # Create LangSmith client
        _langsmith_client = Client(
            api_key=api_key,
            api_url=endpoint
        )
        
        _project_name = project_name
        _tracing_enabled = True
        
        logger.info(f"✅ LangSmith tracing initialized for project: {project_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {e}")
        _tracing_enabled = False
        return False


def get_tracing_callback() -> Optional[Any]:
    """
    Get a LangChain tracing callback for manual instrumentation.
    
    Returns:
        LangChainTracer callback if tracing is enabled, None otherwise
        
    Example:
        >>> callback = get_tracing_callback()
        >>> if callback:
        ...     llm.invoke(prompt, callbacks=[callback])
    """
    if not _tracing_enabled or not LANGSMITH_AVAILABLE or not LangChainTracer:
        return None
    
    try:
        return LangChainTracer(project_name=_project_name)
    except Exception as e:
        logger.warning(f"Failed to create tracing callback: {e}")
        return None


def create_run_config(
    thread_id: Optional[str] = None,
    user_query: Optional[str] = None,
    target_location: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a configuration dict for graph invocation with tracing metadata.
    
    This function creates a properly structured config dict that includes
    both LangGraph's thread_id configuration and LangSmith metadata for
    better trace filtering and organization.
    
    Args:
        thread_id: Conversation thread ID (generated if not provided)
        user_query: User's query (truncated for metadata)
        target_location: Target location being queried
        tags: Custom tags for filtering in LangSmith
        metadata: Additional metadata to include
        
    Returns:
        Config dict for graph.ainvoke()
        
    Example:
        >>> config = create_run_config(
        ...     user_query="What's the best time to visit Sigiriya?",
        ...     target_location="Sigiriya",
        ...     tags=["tourism_query"]
        ... )
        >>> result = await graph.ainvoke(state, config)
    """
    # Build tags
    run_tags = ["travion", "rag-agent"]
    if tags:
        run_tags.extend(tags)
    if target_location:
        run_tags.append(f"location:{target_location}")
    
    # Build metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "project": "travion-ai-engine",
    }
    if user_query:
        run_metadata["user_query"] = user_query[:100]  # Truncate for metadata
    if target_location:
        run_metadata["target_location"] = target_location
    if metadata:
        run_metadata.update(metadata)
    
    return {
        "configurable": {
            "thread_id": thread_id or str(uuid.uuid4())
        },
        "tags": run_tags,
        "metadata": run_metadata,
    }


def trace_node(node_name: str, run_type: str = "chain") -> Callable[[F], F]:
    """
    Decorator to trace a LangGraph node with LangSmith.
    
    This decorator wraps async node functions to:
    1. Create a LangSmith trace span
    2. Log input state and output state
    3. Track execution time
    4. Capture errors
    
    Args:
        node_name: Name of the node (shown in LangSmith UI)
        run_type: Type of run ("chain", "llm", "tool", "retriever")
        
    Returns:
        Decorated function with tracing
        
    Example:
        >>> @trace_node("router")
        >>> async def router_node(state: GraphState, llm=None) -> GraphState:
        ...     # Node logic here
        ...     return updated_state
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = TracingMetrics(node_name=node_name)
            
            # Log node entry
            logger.debug(f"📍 Node '{node_name}' started")
            
            # Extract state info for metadata
            state = args[0] if args else kwargs.get("state", {})
            metrics.metadata = {
                "user_query": state.get("user_query", "")[:50] if isinstance(state, dict) else "",
                "intent": str(state.get("intent", "")) if isinstance(state, dict) else "",
            }
            
            try:
                # Execute the node function
                if LANGSMITH_AVAILABLE and _tracing_enabled and traceable:
                    # Use LangSmith traceable decorator
                    traced_func = traceable(name=node_name, run_type=run_type)(func)
                    result = await traced_func(*args, **kwargs)
                else:
                    # Execute without tracing
                    result = await func(*args, **kwargs)
                
                metrics.complete(success=True)
                
                # Log node completion
                logger.debug(
                    f"✅ Node '{node_name}' completed in {metrics.latency_ms:.2f}ms"
                )
                
                return result
                
            except Exception as e:
                metrics.complete(success=False, error_message=str(e))
                
                # Log error
                logger.error(
                    f"❌ Node '{node_name}' failed after {metrics.latency_ms:.2f}ms: {e}"
                )
                
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def trace_tool(tool_name: str) -> Callable[[F], F]:
    """
    Decorator to trace tool execution with LangSmith.
    
    Similar to trace_node but specifically for tool functions.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Decorated function with tracing
        
    Example:
        >>> @trace_tool("crowdcast")
        >>> async def get_crowd_prediction(location: str, date: str):
        ...     # Tool logic here
        ...     return prediction
    """
    return trace_node(tool_name, run_type="tool")


def trace_retrieval(retriever_name: str) -> Callable[[F], F]:
    """
    Decorator to trace retrieval operations.
    
    Args:
        retriever_name: Name of the retriever
        
    Returns:
        Decorated function with tracing
    """
    return trace_node(retriever_name, run_type="retriever")


def trace_llm(llm_name: str) -> Callable[[F], F]:
    """
    Decorator to trace LLM calls.
    
    Args:
        llm_name: Name/identifier for the LLM call
        
    Returns:
        Decorated function with tracing
    """
    return trace_node(llm_name, run_type="llm")


class NodeTracer:
    """
    Context manager for node tracing.
    
    Use this when you need more control over trace metadata than
    the decorator provides.
    
    Example:
        >>> async with NodeTracer("complex_node") as tracer:
        ...     tracer.add_metadata("step", "initial")
        ...     result = await do_something()
        ...     tracer.add_metadata("step", "final")
        ...     return result
    """
    
    def __init__(self, node_name: str, run_type: str = "chain"):
        self.node_name = node_name
        self.run_type = run_type
        self.metrics = TracingMetrics(node_name=node_name)
        self._run_tree = None
        
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the current trace."""
        self.metrics.metadata[key] = value
        
    async def __aenter__(self):
        logger.debug(f"📍 Trace '{self.node_name}' started")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.metrics.complete(success=False, error_message=str(exc_val))
            logger.error(f"❌ Trace '{self.node_name}' failed: {exc_val}")
        else:
            self.metrics.complete(success=True)
            logger.debug(
                f"✅ Trace '{self.node_name}' completed in {self.metrics.latency_ms:.2f}ms"
            )
        return False  # Don't suppress exceptions


def log_rag_metrics(
    query: str,
    documents_retrieved: int,
    relevance_score: float,
    response_length: int,
    reasoning_loops: int
):
    """
    Log RAG-specific metrics for monitoring dashboard.
    
    Args:
        query: User query
        documents_retrieved: Number of documents retrieved
        relevance_score: Average relevance score
        response_length: Length of generated response
        reasoning_loops: Number of self-correction loops
    """
    logger.info(
        f"📊 RAG Metrics | "
        f"docs={documents_retrieved} | "
        f"relevance={relevance_score:.2f} | "
        f"response_len={response_length} | "
        f"loops={reasoning_loops}"
    )
    
    if _tracing_enabled and _langsmith_client:
        try:
            # Log as custom event (if supported)
            pass  # LangSmith metrics are tracked automatically through traces
        except Exception:
            pass  # Silently ignore metric logging failures
