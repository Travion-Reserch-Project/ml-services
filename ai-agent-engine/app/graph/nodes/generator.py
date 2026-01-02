"""
Generator Node: LLM Response Generation with Context Injection.

This node synthesizes all gathered information (retrieved documents,
web search results, constraint checks) into a coherent response.

Research Pattern:
    Context-Aware Generation - The generator receives structured context
    from multiple sources and produces a response that addresses the
    user's query while respecting detected constraints.

Response Types:
    - GREETING: Casual, friendly response
    - TOURISM_QUERY: Informative response with retrieved knowledge
    - TRIP_PLANNING: Structured itinerary with optimizations
    - OFF_TOPIC: Polite redirect to tourism topics
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

from ..state import GraphState, IntentType, ShadowMonitorLog

logger = logging.getLogger(__name__)

# System prompts for different response types
SYSTEM_PROMPTS = {
    "tourism_guide": """You are Travion, an expert AI tour guide for Sri Lanka.

Your personality:
- Warm and hospitable, reflecting Sri Lankan culture
- Knowledgeable about history, nature, and local customs
- Practical with logistics while inspiring with stories
- Respectful of cultural and religious sensitivities

Guidelines:
- Provide accurate, helpful information about Sri Lankan destinations
- Include practical tips (best times, what to bring, costs)
- Weave in interesting historical or cultural context
- Warn about any constraints (Poya days, crowds, weather)
- Suggest alternatives when plans have issues

Current context will be provided. Use it to give accurate, specific answers.""",

    "trip_planner": """You are Travion, an AI travel planning assistant for Sri Lanka.

When creating itineraries:
1. Consider the optimizations suggested (crowd levels, lighting, events)
2. Include specific times based on constraint analysis
3. Note any warnings (Poya day restrictions, high crowds)
4. Provide practical tips for each stop
5. Format as a clear, structured itinerary

Constraint information will be provided - incorporate it into your plans.""",

    "greeting": """You are Travion, a friendly AI tour guide for Sri Lanka.
Respond warmly to greetings. Offer to help with travel planning or questions about Sri Lanka.
Keep responses concise and inviting.""",

    "off_topic": """You are Travion, an AI tour guide specialized in Sri Lankan tourism.
Politely redirect off-topic questions back to travel and tourism.
Be helpful but clear about your specialty."""
}


def build_context_string(state: GraphState) -> str:
    """
    Build a context string from all available information.

    Args:
        state: Current graph state

    Returns:
        Formatted context string for LLM injection
    """
    parts = []

    # Add retrieved documents
    docs = state.get("retrieved_documents", [])
    if docs:
        parts.append("=== KNOWLEDGE BASE ===")
        for i, doc in enumerate(docs[:5], 1):
            location = doc.get("metadata", {}).get("location", "Unknown")
            parts.append(f"\n[{i}] {location}:")
            parts.append(doc["content"][:400])
        parts.append("")

    # Add web search results
    web_results = state.get("web_search_results", [])
    if web_results:
        parts.append("=== WEB SEARCH RESULTS ===")
        for r in web_results[:3]:
            parts.append(f"\n- {r.get('title', 'Untitled')}:")
            parts.append(f"  {r.get('content', '')[:300]}")
        parts.append("")

    # Add constraint information
    constraints = state.get("_constraint_results")
    if constraints:
        parts.append("=== CONSTRAINT ANALYSIS ===")
        parts.append(constraints.get("recommendation", ""))
        parts.append("")

    # Add itinerary if generated
    itinerary = state.get("itinerary")
    if itinerary:
        parts.append("=== OPTIMIZED SCHEDULE ===")
        for slot in itinerary:
            parts.append(f"- {slot['time']}: {slot['location']}")
            parts.append(f"  Crowd: {slot['crowd_prediction']}%, Lighting: {slot['lighting_quality']}")
            if slot.get("notes"):
                parts.append(f"  Note: {slot['notes']}")
        parts.append("")

    return "\n".join(parts)


def get_system_prompt(intent: Optional[IntentType]) -> str:
    """Get appropriate system prompt based on intent."""
    if intent == IntentType.GREETING:
        return SYSTEM_PROMPTS["greeting"]
    elif intent == IntentType.OFF_TOPIC:
        return SYSTEM_PROMPTS["off_topic"]
    elif intent == IntentType.TRIP_PLANNING:
        return SYSTEM_PROMPTS["trip_planner"]
    else:
        return SYSTEM_PROMPTS["tourism_guide"]


async def generator_node(state: GraphState, llm=None) -> GraphState:
    """
    Generator Node: Produce final response using LLM.

    This node synthesizes all gathered context into a coherent response.
    It handles different intents appropriately and incorporates constraint
    information when relevant.

    Args:
        state: Current graph state
        llm: LangChain LLM instance

    Returns:
        Updated GraphState with generated response

    Research Note:
        The generator implements "Grounded Generation" - it strictly uses
        provided context to reduce hallucination risk while maintaining
        natural, helpful responses.
    """
    query = state["user_query"]
    intent = state.get("intent", IntentType.TOURISM_QUERY)

    logger.info(f"Generator processing intent: {intent.value}")

    # Build context from all sources
    context = build_context_string(state)

    # Get appropriate system prompt
    system_prompt = get_system_prompt(intent)

    # Build user message with context
    if context and intent not in [IntentType.GREETING, IntentType.OFF_TOPIC]:
        user_message = f"""Context Information:
{context}

User Question: {query}

Please provide a helpful response based on the context above."""
    else:
        user_message = query

    # Build conversation history from state messages
    # This enables multi-turn conversation memory
    messages = state.get("messages", [])
    conversation_messages = [{"role": "system", "content": system_prompt}]

    # Add previous messages from history (excluding the current query)
    # Messages are stored as [user, assistant, user, assistant, ...]
    # We include previous turns to maintain conversation context
    if len(messages) > 1:
        # Include up to last 10 messages (5 turns) for context
        history_messages = messages[:-1][-10:]
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                conversation_messages.append({"role": role, "content": content})

    # Add current user message (with context if applicable)
    conversation_messages.append({"role": "user", "content": user_message})

    # Generate response
    if llm:
        try:
            response = await llm.ainvoke(conversation_messages)
            generated_text = response.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            generated_text = generate_fallback_response(state)
    else:
        generated_text = generate_fallback_response(state)

    # Log generation
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="generator",
        input_context={
            "intent": intent.value,
            "context_length": len(context),
            "has_constraints": bool(state.get("constraint_violations"))
        },
        result="generated",
        details=f"Response length: {len(generated_text)} chars",
        action_taken=None
    )

    # Add assistant message to conversation history for future turns
    # This enables the conversation memory to persist across messages
    assistant_message = {"role": "assistant", "content": generated_text}

    return {
        **state,
        "generated_response": generated_text,
        "messages": [assistant_message],  # Will be appended via operator.add
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry]
    }


def generate_fallback_response(state: GraphState) -> str:
    """
    Generate a response without LLM (rule-based fallback).

    Args:
        state: Current graph state

    Returns:
        Fallback response string
    """
    intent = state.get("intent", IntentType.TOURISM_QUERY)
    target_location = state.get("target_location")

    if intent == IntentType.GREETING:
        return ("Ayubowan! Welcome to Travion, your AI guide to Sri Lanka. "
                "I can help you plan trips, find destinations, and learn about "
                "Sri Lankan culture. What would you like to explore today?")

    elif intent == IntentType.OFF_TOPIC:
        return ("I'm Travion, specialized in Sri Lankan tourism. "
                "I'd be happy to help you explore Sri Lanka's amazing "
                "destinations, from ancient ruins to pristine beaches. "
                "What would you like to know about Sri Lanka?")

    elif intent == IntentType.TRIP_PLANNING:
        # Generate basic itinerary from state
        itinerary = state.get("itinerary", [])
        constraints = state.get("_constraint_results", {})

        response_parts = [f"Here's an optimized plan for visiting {target_location or 'Sri Lanka'}:"]

        if constraints.get("recommendation"):
            response_parts.append(f"\n{constraints['recommendation']}")

        if itinerary:
            response_parts.append("\nSuggested Schedule:")
            for slot in itinerary:
                response_parts.append(f"- {slot['time']}: {slot['location']}")
                if slot.get("notes"):
                    response_parts.append(f"  ({slot['notes']})")

        return "\n".join(response_parts)

    else:
        # Tourism query - use retrieved docs
        docs = state.get("retrieved_documents", [])
        if docs:
            best_doc = docs[0]
            location = best_doc.get("metadata", {}).get("location", target_location or "Sri Lanka")
            return (f"Here's what I found about {location}:\n\n"
                    f"{best_doc['content'][:500]}...")
        else:
            return (f"I'd be happy to tell you more about "
                    f"{target_location or 'Sri Lanka'}. Could you be more "
                    "specific about what you'd like to know?")
