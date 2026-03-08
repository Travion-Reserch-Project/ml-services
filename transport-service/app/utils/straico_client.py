"""
Straico API Client

Handles communication with Straico API for LLM inference.
Includes error handling, retries, and token management.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from app.schemas.knowledge import StraicoRequest, StraicoResponse

logger = logging.getLogger(__name__)


class StraicoAPIError(Exception):
    """Custom exception for Straico API errors"""
    pass


class StraicoClient:
    """
    Client for interacting with Straico API
    
    Features:
    - Async and sync support
    - Automatic retries with exponential backoff
    - Token counting and management
    - Error handling and logging
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize Straico client
        
        Args:
            api_key: Straico API key (defaults to STRAICO_API_KEY env var)
            base_url: API base URL (defaults to STRAICO_BASE_URL env var)
            model: Default model to use (defaults to STRAICO_MODEL env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("STRAICO_API_KEY")
        self.base_url = base_url or os.getenv(
            "STRAICO_BASE_URL", 
            "https://api.straico.com/v1"
        )
        self.model = model or os.getenv(
            "STRAICO_MODEL", 
            "anthropic/claude-3.7-sonnet"
        )
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError(
                "Straico API key not found. Set STRAICO_API_KEY environment variable."
            )
        
        # HTTP client
        self.client = httpx.Client(
            timeout=self.timeout,
            headers=self._get_headers()
        )
        
        self.async_client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self._get_headers()
        )
        
        logger.info(f"✓ Straico client initialized (model: {self.model})")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _build_messages(
        self, 
        query: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build messages array for API request
        
        Args:
            query: User query
            context: Retrieved context from vector search
            system_prompt: System prompt override
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # System message
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # User message with context
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_content = query
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, StraicoAPIError)),
        reraise=True
    )
    def generate(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> StraicoResponse:
        """
        Generate completion using Straico API (synchronous)
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: System prompt
            model: Model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            StraicoResponse with generated text
        """
        try:
            messages = self._build_messages(query, context, system_prompt)
            
            selected_model = model or self.model
            
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                # Straico chat endpoint expects a models array.
                "models": [selected_model]
            }
            
            logger.debug(f"Sending request to Straico API: {selected_model}")
            
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            
            response.raise_for_status()
            data = response.json()

            # Straico response format: data.completions.<model>.completion
            # Fallback to OpenAI-style choices if present.
            if "data" in data and isinstance(data["data"], dict):
                completions = data["data"].get("completions", {})
                model_key = selected_model if selected_model in completions else next(iter(completions), None)
                if model_key:
                    completion = completions[model_key].get("completion", {})
                    choices = completion.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        finish_reason = choices[0].get("finish_reason")
                        tokens_used = completion.get("usage", {}).get("total_tokens")
                    else:
                        raise StraicoAPIError("No choices in Straico completion")
                else:
                    raise StraicoAPIError("No completion payload found in Straico response")
            elif "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                finish_reason = data["choices"][0].get("finish_reason")
                tokens_used = data.get("usage", {}).get("total_tokens")
            else:
                raise StraicoAPIError("Invalid response format from Straico API")
            
            logger.info(f"✓ Generated response ({tokens_used} tokens)")
            
            return StraicoResponse(
                content=content,
                model=selected_model,
                tokens_used=tokens_used,
                finish_reason=finish_reason
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Straico API HTTP error: {e.response.status_code} - {e.response.text}")
            raise StraicoAPIError(f"API request failed: {e.response.text}")
        
        except httpx.RequestError as e:
            logger.error(f"Straico API request error: {e}")
            raise StraicoAPIError(f"Request failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in Straico client: {e}")
            raise
    
    async def generate_async(
        self,
        query: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> StraicoResponse:
        """
        Generate completion using Straico API (asynchronous)
        
        Same arguments as generate()
        """
        try:
            messages = self._build_messages(query, context, system_prompt)
            
            selected_model = model or self.model
            
            payload = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "models": [selected_model]
            }
            
            logger.debug(f"Sending async request to Straico API: {selected_model}")
            
            response = await self.async_client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            
            response.raise_for_status()
            data = response.json()

            if "data" in data and isinstance(data["data"], dict):
                completions = data["data"].get("completions", {})
                model_key = selected_model if selected_model in completions else next(iter(completions), None)
                if model_key:
                    completion = completions[model_key].get("completion", {})
                    choices = completion.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        finish_reason = choices[0].get("finish_reason")
                        tokens_used = completion.get("usage", {}).get("total_tokens")
                    else:
                        raise StraicoAPIError("No choices in Straico completion")
                else:
                    raise StraicoAPIError("No completion payload found in Straico response")
            elif "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                finish_reason = data["choices"][0].get("finish_reason")
                tokens_used = data.get("usage", {}).get("total_tokens")
            else:
                raise StraicoAPIError("Invalid response format from Straico API")
            
            logger.info(f"✓ Generated async response ({tokens_used} tokens)")
            
            return StraicoResponse(
                content=content,
                model=selected_model,
                tokens_used=tokens_used,
                finish_reason=finish_reason
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Straico API HTTP error: {e.response.status_code}")
            raise StraicoAPIError(f"API request failed: {e.response.text}")
        
        except Exception as e:
            logger.error(f"Unexpected error in async Straico client: {e}")
            raise
    
    
    def close(self):
        """Close HTTP clients"""
        self.client.close()
    
    async def aclose(self):
        """Close async HTTP client"""
        await self.async_client.aclose()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.aclose()


# ================================================
# Utility Functions
# ================================================

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (4 chars ≈ 1 token for English)
    For more accurate counting, use tiktoken
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def truncate_context(context: str, max_tokens: int = 3000) -> str:
    """
    Truncate context to fit within token limit
    
    Args:
        context: Context text
        max_tokens: Maximum tokens allowed
        
    Returns:
        Truncated context
    """
    estimated_tokens = estimate_tokens(context)
    
    if estimated_tokens <= max_tokens:
        return context
    
    # Truncate to approximate character count
    max_chars = max_tokens * 4
    truncated = context[:max_chars]
    
    logger.warning(f"Context truncated from {estimated_tokens} to ~{max_tokens} tokens")
    
    return truncated + "\n\n[Context truncated...]"


# ================================================
# Testing
# ================================================

if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    try:
        client = StraicoClient()
        
        response = client.generate(
            query="What is the capital of Sri Lanka?",
            context="Sri Lanka's capital is Colombo, but the administrative capital is Sri Jayawardenepura Kotte.",
            temperature=0.5,
            max_tokens=100
        )
        
        print(f"\n{'='*60}")
        print(f"Generated Response:")
        print(f"{'='*60}")
        print(response.content)
        print(f"\nModel: {response.model}")
        print(f"Tokens: {response.tokens_used}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
