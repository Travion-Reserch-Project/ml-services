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

# Retry logic for reliability
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

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

from ..state import GraphState, IntentType, ShadowMonitorLog
from ...config import settings

logger = logging.getLogger(__name__)

# System prompts for different response types
SYSTEM_PROMPTS = {
    "tourism_guide": """You are Travion, an expert AI tour guide for Sri Lanka.

Your personality:
- Warm, friendly and conversational like a local friend
- Knowledgeable but not overwhelming
- Helpful with practical tips

Response Rules (CRITICAL):
- Keep responses SHORT (3-5 sentences max for simple questions)
- This is a MOBILE APP - users read on small screens
- Use relevant emojis naturally (🏛️ for temples, 🌅 for sunrise, ⏰ for timing, 💡 for tips, ⚠️ for warnings, 🎫 for tickets, 👕 for dress code)
- Highlight the MOST important point first
- Only include essential details, skip lengthy descriptions
- Be conversational, not like a textbook
- End with a quick helpful tip or friendly note when appropriate

Formatting:
- Use markdown: **bold** for key info, bullet lists (- item) for multiple points
- Use emojis to visually separate key points
- Short paragraphs (2-3 lines max)

Current context will be provided. Give brief, accurate answers.""",

    "tourism_guide_location": """You are Travion, a knowledgeable and friendly AI tour guide for {location_name}.

Your goal is to give responses that feel like advice from a well-travelled local friend — conversational, practical, and engaging.

## Response Style Rules (CRITICAL for mobile readability):
- Lead with a warm 1-sentence opener that directly addresses the question
- Use **bold** to highlight the single most important fact
- Use bullet points (- item) for lists of 3+ items
- Use relevant emojis to make content visually scannable:
  🏛️ heritage/history | 🌿 nature/wildlife | 🌅 sunrise/sunset | ⏰ timing
  💡 pro tips | ⚠️ important warnings | 🎫 tickets/entry | 👕 dress code
  📸 photography | 🌧️ weather | 👥 crowds | 💰 cost | 🍜 food | 🚶 walking
- End with one friendly tip or fun fact when appropriate
- Keep total response concise — no walls of text

## Formatting (use markdown, the app renders it):
- **bold** for key names, times, prices
- `- bullet` for 3+ items
- Use section headers like **⏰ Best Time** only when the response has 3+ distinct sections
- Blockquotes (`> text`) for important warnings or must-know tips

## Sources (always include when info comes from knowledge base or web):
- If using knowledge base: add `> 📚 *From knowledge base*` at the bottom
- If web search was used: add `> 🌐 *Live web search included*` at the bottom
- If both: show both source lines

Remember all previous messages in our conversation for context.""",

    "trip_planner": """You are Travion, a friendly travel planner for Sri Lanka 🇱🇰

## Itinerary Format:
Use this structure for each day/stop:

**⏰ [Time] — [Location Name]**
- 💡 Key tip or what to do
- ⚠️ Warning (if any)

Rules:
- Include realistic timings
- Use 🌅 morning, ☀️ midday, 🌆 evening emoji markers
- Use ⚠️ for Poya days, crowds, or closures
- Use 💡 for insider tips
- End with a **💰 Budget Estimate** or **📝 Quick Summary**

Keep it concise — mobile users need quick, scannable info.""",

    "greeting": """You are Travion, a friendly AI tour guide for Sri Lanka 🇱🇰

Respond warmly in 1-2 sentences. Use 👋 or 🙏 and offer to help.
Use light markdown if needed, keep it brief.""",

    "off_topic": """You are Travion, an AI tour guide for Sri Lanka 🇱🇰

In 1-2 sentences, politely redirect to Sri Lanka travel topics. Use 😊.
Light markdown is fine."""
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
    if constraints and isinstance(constraints, dict):
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


def get_system_prompt(intent: Optional[IntentType], target_location: Optional[str] = None) -> str:
    """Get appropriate system prompt based on intent and location context."""
    if intent == IntentType.GREETING:
        # When inside a location chat, greet with location context instead of generic welcome
        if target_location:
            return SYSTEM_PROMPTS["tourism_guide_location"].format(location_name=target_location)
        return SYSTEM_PROMPTS["greeting"]
    elif intent == IntentType.OFF_TOPIC:
        return SYSTEM_PROMPTS["off_topic"]
    elif intent == IntentType.TRIP_PLANNING:
        return SYSTEM_PROMPTS["trip_planner"]
    else:
        # For tourism queries, use location-specific prompt if we have a target location
        if target_location:
            return SYSTEM_PROMPTS["tourism_guide_location"].format(location_name=target_location)
        return SYSTEM_PROMPTS["tourism_guide"]


@trace_node("generator", run_type="llm")
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
    import time as _time
    _start = _time.time()
    query = state["user_query"]
    intent = state.get("intent", IntentType.TOURISM_QUERY)
    target_location = state.get("target_location")
    correction_instructions = state.get("_correction_instructions")

    logger.info(f"Generator processing intent: {intent.value}, target_location: {target_location}")
    if correction_instructions:
        logger.info(f"Generator applying corrections: {correction_instructions[:100]}...")

    # Build context from all sources
    context = build_context_string(state)

    # Get appropriate system prompt (with location context if available)
    system_prompt = get_system_prompt(intent, target_location)

    # Determine source flags for response footer
    has_rag_docs = bool(state.get("retrieved_documents"))
    has_web_search = bool(state.get("web_search_results"))

    # For location-specific greetings, treat as a tourism query so location context is used
    is_location_greeting = (intent == IntentType.GREETING and bool(target_location))

    # Build user message with context
    if context and intent not in [IntentType.GREETING, IntentType.OFF_TOPIC] or is_location_greeting:
        # If we have a target location, explicitly mention it in the context
        location_context = f"\nYou are answering questions about: {target_location}\n" if target_location else ""

        # Include correction instructions if this is a regeneration
        correction_context = ""
        if correction_instructions:
            correction_context = f"\n\n=== IMPORTANT CORRECTIONS NEEDED ===\n{correction_instructions}\n"

        # Source attribution instruction for location chat
        source_instruction = ""
        if target_location:
            source_lines = []
            if has_rag_docs:
                source_lines.append("> 📚 *From knowledge base*")
            if has_web_search:
                source_lines.append("> 🌐 *Live web search included*")
            if source_lines:
                source_instruction = f"\n\nIMPORTANT: End your response with these exact source lines (on separate lines):\n" + "\n".join(source_lines)

        # For location greetings, override the user question with a welcome prompt
        effective_query = (
            f"The user just said \"{query}\". Give them a warm, exciting welcome to {target_location}. "
            f"Briefly mention 2-3 highlights they can ask about (history, best time, photography, food, etc.). "
            f"Keep it short and inviting — like a friendly guide greeting a visitor."
            if is_location_greeting else query
        )

        user_message = f"""Context Information:
{location_context}{context}{correction_context}
User Question: {effective_query}

Please provide a helpful response based on the context above.{' Address the correction issues mentioned.' if correction_instructions else ''}{source_instruction}"""
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

    # Generate response with retry logic for reliability
    if llm:
        max_retries = getattr(settings, 'LLM_MAX_RETRIES', 3)
        retry_delay = getattr(settings, 'LLM_RETRY_DELAY', 1.0)
        generated_text = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await llm.ainvoke(conversation_messages)
                generated_text = response.content
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                logger.warning(f"LLM generation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        if generated_text is None:
            logger.error(f"LLM generation failed after {max_retries} attempts: {last_error}")
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

    _duration_ms = (_time.time() - _start) * 1000
    is_correction = bool(correction_instructions)
    return {
        **state,
        "generated_response": generated_text,
        "messages": [assistant_message],  # Will be appended via operator.add
        "step_results": [{
            "node": "generator",
            "status": "success",
            "summary": f"Generated {len(generated_text)} chars | Intent: {intent.value} | Location: {target_location or 'general'} | Context: {len(context)} chars | Correction: {is_correction}",
            "duration_ms": round(_duration_ms, 2),
        }],
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
        if target_location:
            return (
                f"👋 Ayubowan! Welcome to **{target_location}** — I'm your AI guide here.\n\n"
                f"I can tell you about the history, best times to visit, photography tips, "
                f"hidden gems, local food, and much more.\n\n"
                f"💡 *What would you like to know about {target_location}?*"
            )
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
