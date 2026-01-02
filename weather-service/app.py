from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import logging
from dotenv import load_dotenv

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# FastAPI app initialization
# -------------------------------------------------
app = FastAPI(
    title="Weather Safety Service API",
    description="UV & weather-based risk prediction service for tourists",
    version="1.0.0"
)

# Load environment variables
load_dotenv()

# -------------------------------------------------
# CORS configuration
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Lazy-loaded state
# -------------------------------------------------
weather_service = None
models_downloaded = False


# -------------------------------------------------
# Ensure models are downloaded
# -------------------------------------------------
def ensure_models_downloaded():
    global models_downloaded

    if not models_downloaded:
        logger.info("=" * 60)
        logger.info("Checking for weather ML models...")
        logger.info("=" * 60)

        try:
            from utils.model_downloader import download_weather_models
            downloaded = download_weather_models()

            if downloaded:
                logger.info(f"✓ Prepared {len(downloaded)} model artifact(s)")
            else:
                logger.warning("⚠️ No models downloaded – running in limited mode")

        except Exception as e:
            logger.error(f"✗ Failed to download models: {e}")
            logger.warning("⚠️ Service starting without trained model")

        models_downloaded = True
        logger.info("=" * 60)


# -------------------------------------------------
# Lazy-load inference service
# -------------------------------------------------
def get_weather_service():
    global weather_service

    if weather_service is None:
        from utils.weather_inference import WeatherRiskService

        model_path = os.getenv(
            "WEATHER_MODEL_FILE",
            "models/uv_risk_model.pkl"
        )

        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Model file not found at {model_path}")

        weather_service = WeatherRiskService(model_path)

    return weather_service


# -------------------------------------------------
# Request schemas
# -------------------------------------------------
class WeatherFeatures(BaseModel):
    Skin_Type: int = Field(..., alias="Skin Type", ge=1, le=6)
    UV_Index: float = Field(..., alias="UV Index", ge=0, le=15)
    Time_of_Day: int = Field(..., alias="Time of Day", ge=0, le=23)
    Historical_Sunburn: int = Field(..., alias="Historical Sunburn", ge=0, le=100)
    Historical_Tanning: int = Field(..., alias="Historical Tanning", ge=0, le=100)
    Skin_Product_Interaction: str = Field(..., alias="Skin Product Interaction")
    Use_of_Sunglasses: str = Field(..., alias="Use of Sunglasses/Hat/Shade")
    Cloud_Cover: str = Field(..., alias="Cloud Cover")
    Age: int = Field(..., ge=16, le=120)
    Temperature_C: float
    Humidity_pct: float = Field(..., alias="Humidity_%")

    class Config:
        allow_population_by_field_name = True

class PredictRequest(BaseModel):
    features: WeatherFeatures
    user_location: Optional[str] = None


class BatchPredictRequest(BaseModel):
    features_list: List[WeatherFeatures]


# -------------------------------------------------
# Root endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Weather Safety Service API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/weather/predict",
            "batch_predict": "/api/weather/batch_predict",
            "health": "/api/weather/health"
        }
    }


# -------------------------------------------------
# Predict single instance
# -------------------------------------------------
@app.post("/api/weather/predict")
def predict_weather_risk(request: PredictRequest):
    ensure_models_downloaded()

    try:
        service = get_weather_service()

        if service is None:
            return {"success": False, "error": "Weather service unavailable"}

        result = service.predict_one(request.features.dict())

        return {
            "success": True,
            "prediction": result,
            "location": request.user_location
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# Batch prediction
# -------------------------------------------------
@app.post("/api/weather/batch_predict")
def batch_predict_weather_risk(request: BatchPredictRequest):
    ensure_models_downloaded()

    try:
        service = get_weather_service()

        if service is None:
            return {"success": False, "error": "Weather service unavailable"}

        items = [f.dict() for f in request.features_list]
        results = service.predict_batch(items)

        return {
            "success": True,
            "predictions": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# HEALTH CHECK (EXACT FORMAT YOU REQUESTED)
# -------------------------------------------------
@app.get("/api/weather/health")
def health_check():
    ensure_models_downloaded()
    service = get_weather_service()

    status = service.status() if service else {}

    model_path = os.getenv(
        "WEATHER_MODEL_FILE",
        "models/uv_risk_gradient_boosting.pkl"
    )

    model_exists = os.path.exists(model_path)

    return {
        "status": "healthy",
        "model_loaded": status.get("model") == "loaded",
        "scaler_loaded": status.get("scaler") == "loaded",
        "model_file_exists": model_exists,
        "feature_count": status.get("feature_count"),
        "targets": ["low", "moderate", "high", "very high"]
    }


# -------------------------------------------------
# Run locally
# -------------------------------------------------
# uvicorn app:app --reload --port 8004
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
