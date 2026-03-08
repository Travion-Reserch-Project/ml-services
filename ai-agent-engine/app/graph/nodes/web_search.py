"""
Web Search Node: Real-Time Information Fallback.

This node is triggered when the Grader determines that local knowledge
is insufficient. It uses Tavily to search for real-time information.

Research Pattern:
    Retrieval-Augmented Generation with Fallback - When the primary
    knowledge source (ChromaDB) lacks information, the system
    transparently falls back to web search.

Use Cases:
    - Current weather conditions
    - Recent price changes
    - New attractions or closures
    - Events not in the calendar database
"""

import logging
import time as _time_mod
from datetime import datetime
from typing import Dict, Optional

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

from ..state import GraphState, ShadowMonitorLog
from ...tools.web_search import get_web_search_tool
from ...config import settings

logger = logging.getLogger(__name__)


@trace_node("web_search", run_type="tool")
async def web_search_node(state: GraphState) -> GraphState:
    """
    Web Search Node: Fetch real-time information from the web.

    This node is triggered by the grader when local knowledge is insufficient.
    It performs a targeted web search and adds results to the context.

    Args:
        state: Current graph state

    Returns:
        Updated GraphState with web search results

    Research Note:
        The web search is transparent to the user - the agent acknowledges
        when it needs to look up external information, maintaining trust.
    """
    query = state["user_query"]
    target_location = state.get("target_location")

    _start = _time_mod.time()
    logger.info(f"Web Search triggered for: {query[:50]}...")

    # Get web search tool
    web_search = get_web_search_tool(api_key=settings.TAVILY_API_KEY)

    # Build search query with location context
    search_query = query
    if target_location:
        search_query = f"{target_location} {query}"

    # Perform search
    results = web_search.search_tourism(search_query)

    # Log the search
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="web_search",
        input_context={
            "query": query,
            "search_query": search_query,
            "target_location": target_location
        },
        result="success" if results.get("success") else "fallback",
        details=f"Found {len(results.get('results', []))} results",
        action_taken="augment_context"
    )

    # Format results for context
    web_context = []
    if results.get("results"):
        for r in results["results"][:3]:
            web_context.append({
                "title": r.get("title", ""),
                "content": r.get("content", "")[:500],
                "url": r.get("url", ""),
                "source": "web"
            })

    _duration_ms = (_time_mod.time() - _start) * 1000
    return {
        **state,
        "web_search_results": web_context,
        "step_results": [{
            "node": "web_search",
            "status": "success" if web_context else "warning",
            "summary": f"Found {len(web_context)} web results | Query: {search_query[:60]} | Source: Tavily",
            "duration_ms": round(_duration_ms, 2),
        }],
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry]
    }
