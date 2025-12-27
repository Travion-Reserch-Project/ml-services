"""
Web Search Tool: Real-Time Information Fallback via Tavily.

This module provides web search capability when local knowledge base
doesn't contain sufficient information for a query.

Use Cases:
    - Current weather conditions
    - Recent news or events
    - Real-time prices and availability
    - Information not in the vector database

Research Note:
    The web search is a fallback mechanism triggered by the Grader node
    when retrieved documents are deemed insufficient. This implements
    the "Retrieval-Augmented Generation with Fallback" pattern.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("Tavily not available. Web search disabled.")


class WebSearchTool:
    """
    Web search interface for real-time information retrieval.

    This class wraps the Tavily API to provide context-aware web searches
    focused on Sri Lankan tourism information.

    Attributes:
        client: TavilyClient instance
        search_depth: "basic" or "advanced"
        max_results: Number of results to return

    Research Note:
        Tavily is chosen for its AI-optimized search results that
        return clean, structured content suitable for RAG applications.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_depth: str = "basic",
        max_results: int = 5
    ):
        """
        Initialize Web Search Tool.

        Args:
            api_key: Tavily API key
            search_depth: "basic" or "advanced"
            max_results: Maximum number of results
        """
        self.client = None
        self.search_depth = search_depth
        self.max_results = max_results
        self.enabled = False

        if TAVILY_AVAILABLE and api_key:
            try:
                self.client = TavilyClient(api_key=api_key)
                self.enabled = True
                logger.info("WebSearchTool initialized with Tavily")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily: {e}")

    def search(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform a web search for the given query.

        Args:
            query: Search query
            include_domains: List of domains to search within
            exclude_domains: List of domains to exclude

        Returns:
            Dict with search results

        Example:
            >>> search_tool = WebSearchTool(api_key="...")
            >>> results = search_tool.search("Sri Lanka weather December 2026")
            >>> for result in results["results"]:
            ...     print(result["title"])
        """
        if not self.enabled:
            return self._fallback_response(query)

        try:
            # Add Sri Lanka context to query
            enhanced_query = f"{query} Sri Lanka tourism"

            # Default tourism-related domains
            tourism_domains = include_domains or [
                "tripadvisor.com",
                "lonelyplanet.com",
                "srilanka.travel",
                "booking.com",
                "viator.com"
            ]

            response = self.client.search(
                query=enhanced_query,
                search_depth=self.search_depth,
                max_results=self.max_results,
                include_domains=tourism_domains,
                exclude_domains=exclude_domains or []
            )

            return {
                "success": True,
                "query": query,
                "enhanced_query": enhanced_query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0)
                    }
                    for r in response.get("results", [])
                ],
                "answer": response.get("answer"),
                "source": "tavily",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return self._fallback_response(query, error=str(e))

    def search_tourism(self, query: str) -> Dict:
        """
        Tourism-specific search with curated domains.

        Args:
            query: Search query

        Returns:
            Dict with tourism-focused results
        """
        tourism_domains = [
            "tripadvisor.com",
            "lonelyplanet.com",
            "srilanka.travel",
            "booking.com",
            "viator.com",
            "getyourguide.com",
            "sltda.gov.lk"
        ]

        return self.search(query, include_domains=tourism_domains)

    def search_weather(self, location: str) -> Dict:
        """
        Search for current weather information.

        Args:
            location: Location name in Sri Lanka

        Returns:
            Dict with weather results
        """
        query = f"{location} weather forecast"
        weather_domains = [
            "weather.com",
            "accuweather.com",
            "timeanddate.com",
            "meteo.gov.lk"
        ]

        return self.search(query, include_domains=weather_domains)

    def search_events(self, date_range: str) -> Dict:
        """
        Search for events and festivals.

        Args:
            date_range: Date range description (e.g., "May 2026")

        Returns:
            Dict with event results
        """
        query = f"Sri Lanka festivals events {date_range}"
        event_domains = [
            "srilanka.travel",
            "lonelyplanet.com",
            "tripadvisor.com",
            "timeout.com"
        ]

        return self.search(query, include_domains=event_domains)

    def _fallback_response(
        self,
        query: str,
        error: Optional[str] = None
    ) -> Dict:
        """
        Return a fallback response when web search is unavailable.

        Args:
            query: Original query
            error: Error message if applicable

        Returns:
            Dict with fallback status
        """
        return {
            "success": False,
            "query": query,
            "results": [],
            "answer": None,
            "source": "fallback",
            "message": error or "Web search not available. Using local knowledge only.",
            "timestamp": datetime.now().isoformat()
        }

    def get_context_for_rag(self, query: str) -> str:
        """
        Get web search results formatted for RAG context injection.

        This method formats search results as a single string suitable
        for adding to the LLM context.

        Args:
            query: Search query

        Returns:
            str: Formatted context string

        Example:
            >>> tool = WebSearchTool(api_key="...")
            >>> context = tool.get_context_for_rag("Sigiriya entry fee 2026")
            >>> # Use in LLM prompt: f"Additional context: {context}"
        """
        results = self.search(query)

        if not results["success"] or not results["results"]:
            return ""

        context_parts = ["[Web Search Results]"]

        for i, r in enumerate(results["results"][:3], 1):
            context_parts.append(
                f"\n{i}. {r['title']}\n   {r['content'][:300]}..."
            )

        if results.get("answer"):
            context_parts.append(f"\n\nSummary: {results['answer']}")

        return "\n".join(context_parts)


# Singleton instance
_web_search_tool: Optional[WebSearchTool] = None


def get_web_search_tool(api_key: Optional[str] = None) -> WebSearchTool:
    """
    Get or create the WebSearchTool singleton.

    Args:
        api_key: Tavily API key (only used on first call)

    Returns:
        WebSearchTool: Singleton instance
    """
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool(api_key)
    return _web_search_tool
