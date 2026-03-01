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
    description="AI-powered transport recommendation service using ML classifier",
    version="1.0.0"
)

# Load ML models on startup
logger.info("Initializing transport service...")
ModelService.load_models()

# Register all routes
register_routes(app)

# Run with: uvicorn app:app --reload --port 8001
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
