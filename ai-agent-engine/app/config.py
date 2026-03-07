"""
Configuration settings for the Agentic Tour Guide AI Engine.

This module defines all configuration parameters using Pydantic Settings,
enabling environment variable overrides for production deployment.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Attributes:
        APP_NAME: Application identifier
        APP_VERSION: Semantic version
        DEBUG: Enable debug mode

        # LLM Configuration
        OLLAMA_BASE_URL: Ollama server endpoint
        OLLAMA_MODEL: Model name for inference
        LLM_TEMPERATURE: Sampling temperature (0.0-1.0)
        LLM_MAX_TOKENS: Maximum response tokens

        # ChromaDB Configuration
        CHROMA_PERSIST_DIR: Path to ChromaDB storage
        CHROMA_COLLECTION_NAME: Vector collection name

        # Tavily Web Search
        TAVILY_API_KEY: API key for web search fallback

        # API Configuration
        API_V1_PREFIX: API version prefix
        HOST: Server host
        PORT: Server port
        
        # LangSmith Monitoring
        LANGCHAIN_TRACING_V2: Enable LangSmith tracing
        LANGCHAIN_API_KEY: LangSmith API key
        LANGCHAIN_PROJECT: Project name in LangSmith
    """

    # Application
    APP_NAME: str = "Travion AI Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # LLM Configuration
    # Set LLM_PROVIDER to "gemini", "openai", or "ollama"
    LLM_PROVIDER: str = "openai"

    # Gemini Configuration (Primary - Free tier available)
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"  # or "gemini-1.5-pro" for better quality

    # OpenAI Configuration (Fallback)
    OPENAI_MODEL: str = "gpt-4o"  # or "gpt-4o" for better quality

    # Ollama Configuration (Local fallback)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Common LLM settings
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    # ChromaDB Configuration (local to ai-engine)
    CHROMA_PERSIST_DIR: str = "./vector_db"
    CHROMA_COLLECTION_NAME: str = "tourism_knowledge"

    # OpenAI API Key (for vector embeddings)
    OPENAI_API_KEY: Optional[str] = None

    # Tavily Web Search (Fallback)
    TAVILY_API_KEY: Optional[str] = None

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Data Paths (local to ai-engine)
    HOLIDAYS_DATA_PATH: str = "./data/holidays_2026.json"
    LOCATIONS_DATA_PATH: str = "./data/locations_metadata.csv"

    # CrowdCast Model Path (local to ai-engine)
    CROWDCAST_MODEL_PATH: str = "./models/crowdcast_model.joblib"
    LABEL_ENCODER_PATH: str = "./models/label_encoder.joblib"

    # Agent Configuration
    MAX_REASONING_LOOPS: int = 3
    RETRIEVAL_TOP_K: int = 5
    RELEVANCE_THRESHOLD: float = 0.7
    MAX_WEB_SEARCH_RESULTS: int = 5

    # LangSmith Monitoring Configuration
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "travion-ai-engine"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    # Reliability Configuration
    LLM_TIMEOUT_SECONDS: int = 30
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_DELAY: float = 1.0
    CHROMADB_TIMEOUT_SECONDS: int = 10

    # Active Guardian API Keys (Weather & News Validation)
    OPENWEATHER_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None

    # MCP (Model Context Protocol) Server Configuration
    MCP_GOOGLE_MAPS_URL: Optional[str] = None
    MCP_GOOGLE_MAPS_KEY: Optional[str] = None
    MCP_YELP_URL: Optional[str] = None
    MCP_YELP_KEY: Optional[str] = None
    MCP_TOURISM_URL: Optional[str] = None
    MCP_TOURISM_KEY: Optional[str] = None

    # Google Maps API Key (direct Places API calls for restaurant search)
    GOOGLE_MAPS_API_KEY: Optional[str] = None

    # Service Reliability Configuration
    STRICT_VALIDATION: bool = Field(
        default=True,
        description="Raise errors on service failures instead of silent degradation"
    )
    REQUIRE_WEATHER_API: bool = Field(
        default=False,
        description="Block execution if weather API unavailable"
    )
    REQUIRE_NEWS_API: bool = Field(
        default=False,
        description="Block execution if news API unavailable"
    )

    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_THRESHOLD: int = Field(
        default=5,
        description="Number of failures before circuit opens"
    )
    CIRCUIT_BREAKER_TIMEOUT: int = Field(
        default=60,
        description="Seconds before attempting recovery after circuit opens"
    )

    # Production Session Management
    SESSION_BACKEND: str = Field(
        default="memory",
        description="Session backend: memory, redis, or mongodb"
    )
    REDIS_URL: Optional[str] = None
    MONGODB_SESSION_URL: Optional[str] = None

    # CORS Configuration (update for production)
    CORS_ORIGINS: str = "*"  # Comma-separated list or * for all

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Allow extra fields in .env


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings factory.

    Returns:
        Settings: Application configuration singleton
    """
    return Settings()


# Global settings instance
settings = get_settings()
