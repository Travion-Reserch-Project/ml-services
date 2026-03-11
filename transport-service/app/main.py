"""
FastAPI Application Entry Point

This module initializes and configures the FastAPI application.
"""

import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from routes import register_routes
from app.services import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Transport Service API",
    description="AI-powered transport recommendation and RAG-based knowledge service",
    version="2.0.0"
)

# Load ML models on startup
logger.info("Initializing transport service...")
ModelService.load_models()

# Register all routes
register_routes(app)

logger.info("✓ Transport service ready")
