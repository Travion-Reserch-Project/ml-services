from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Safety Service API",
    description="Tourist safety risk prediction service using ML (multi-output classifier)",
    version="1.0.0"
)

# Load environment variables from .env if present
load_dotenv()

# Allow CORS for mobile/frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of heavy dependencies
safety_service = None
models_downloaded = False


def ensure_models_downloaded():
    """Download models on first request if not already downloaded."""
    global models_downloaded
    if not models_downloaded:
        logger.info("=" * 60)
        logger.info("Checking for ML models...")
        logger.info("=" * 60)
        
        try:
            from utils.model_downloader import download_safety_models
            
            # Download models from MLflow/DagsHub or GitHub Releases
            downloaded = download_safety_models()
            
            if downloaded:
                logger.info(f"✓ Successfully prepared {len(downloaded)} model(s)")
                models_downloaded = True
            else:
                logger.warning("⚠️ No models were downloaded - service will run in limited mode")
                models_downloaded = True  # Don't keep trying
                
        except Exception as e:
            logger.error(f"✗ Failed to download models: {e}")
            logger.warning("⚠️ Service will start without trained model")
            models_downloaded = True  # Don't keep trying
        
        logger.info("=" * 60)


def get_safety_service():
    """Lazy load safety service."""
    global safety_service
    if safety_service is None:
        from utils.safety_inference import SafetyService
        model_path = os.getenv("SAFETY_MODEL_FILE", 'model/safety_risk_model.pkl')
        
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Model not found at {model_path}")
        
        safety_service = SafetyService(model_path)
    return safety_service


# Request/response schemas
class SafetyFeatures(BaseModel):
    """Input features for safety risk prediction"""
    lat: float
    lon: float
    area_cluster: int
    is_beach: int = Field(..., ge=0, le=1)
    is_crowded: int = Field(..., ge=0, le=1)
    is_tourist_place: int = Field(..., ge=0, le=1)
    is_transit: int = Field(..., ge=0, le=1)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    police_nearby: int = Field(..., ge=0, le=1)


class PredictRequest(BaseModel):
    """Request model for single prediction"""
    features: SafetyFeatures
    user_location: Optional[str] = None


class BatchPredictRequest(BaseModel):
    """Request model for batch predictions"""
    features_list: List[SafetyFeatures]


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "Safety Service API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/safety/predict",
            "batch_predict": "/api/safety/batch_predict",
            "health": "/api/safety/health"
        }
    }


@app.post("/api/safety/predict")
def predict_risk(request: PredictRequest):
    """
    Predict safety risk levels for a single location.
    
    Returns risk levels (Low/Medium/High) for 7 incident categories:
    - harassment, pickpocket, scam, bag_snatching, theft, extortion, money_theft
    """
    # Ensure models are downloaded before processing
    ensure_models_downloaded()
    
    try:
        service = get_safety_service()
        
        if service is None:
            return {
                "success": False,
                "error": "Safety service not available"
            }
        
        result = service.predict_one(request.features.dict())
        
        return {
            "success": True,
            "prediction": result,
            "location": request.user_location
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/safety/batch_predict")
def batch_predict_risk(request: BatchPredictRequest):
    """
    Predict safety risk levels for multiple locations.
    Useful for map visualization of risk zones.
    """
    # Ensure models are downloaded before processing
    ensure_models_downloaded()
    
    try:
        service = get_safety_service()
        
        if service is None:
            return {
                "success": False,
                "error": "Safety service not available"
            }
        
        items = [f.dict() for f in request.features_list]
        result = service.predict_batch(items)
        
        return {
            "success": True,
            "predictions": result,
            "count": len(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/safety/health")
def health_check():
    """Detailed health check with model and data status."""
    ensure_models_downloaded()
    service = get_safety_service()
    
    status = service.status() if service else {"model": "not_loaded"}
    
    # Check if model file exists
    model_path = os.getenv("SAFETY_MODEL_FILE", 'model/safety_risk_model.pkl')
    model_exists = os.path.exists(model_path)
    
    return {
        "status": "healthy",
        "model_loaded": status.get("model") == "loaded",
        "scaler_loaded": status.get("scaler") == "loaded",
        "model_file_exists": model_exists,
        "feature_count": status.get("feature_count"),
        "targets": status.get("targets"),
        "note": "Model will be added via DagsHub/MLflow or GitHub Releases after training" if not model_exists else None
    }


# Include modular routes (if we split further later)
from routes.safety_routes import router as safety_router
app.include_router(safety_router, prefix="/api/safety", tags=["safety-extended"])


# Run with: uvicorn app:app --reload --port 8003
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
