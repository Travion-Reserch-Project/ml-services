# Predict Fitzpatrick skin type from a face image
# Predict UV / weather-based risk using an ML model
# Expose these capabilities via REST APIs for your mobile app
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import torch
import logging
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from utils.fitzpatrick import preprocess_image, load_model

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# -------------------------------------------------
# Global model state (populated at startup)
# -------------------------------------------------
fitzpatrick_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weather_service = None


# -------------------------------------------------
# Lifespan: runs model loading ONCE at server start
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    All code before `yield` runs at startup.
    All code after `yield` runs at shutdown.
    """
    global fitzpatrick_model, weather_service

    logger.info("=" * 60)
    logger.info("🚀 Server starting – loading ML models...")
    logger.info("=" * 60)

    # ── 1. Fitzpatrick CNN ──────────────────────────────────────
    try:
        fitzpatrick_model = load_model()
        fitzpatrick_model.to(device)
        fitzpatrick_model.eval()
        logger.info("✓ Fitzpatrick skin-type model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load Fitzpatrick model: {e}")

    # ── 2. Weather risk model (download if needed, then load) ───
    logger.info("-" * 60)
    logger.info("Checking / downloading weather ML models...")
    try:
        from utils.model_downloader import download_weather_models
        downloaded = download_weather_models()
        if downloaded:
            logger.info(f"✓ Prepared {len(downloaded)} model artifact(s)")
        else:
            logger.warning("⚠️  No models downloaded – running in limited mode")
    except Exception as e:
        logger.error(f"✗ Failed to download weather models: {e}")

    try:
        from utils.weather_inference import WeatherRiskService
        model_path = os.getenv("WEATHER_MODEL_FILE", "models/uv_risk_model.pkl")
        if not os.path.exists(model_path):
            logger.warning(f"⚠️  Weather model file not found at {model_path}")
        weather_service = WeatherRiskService(model_path)
        logger.info("✓ Weather risk service loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load WeatherRiskService: {e}")

    logger.info("=" * 60)
    logger.info("✅ All models ready – server is accepting requests")
    logger.info("=" * 60)

    yield  # ← server is live here

    # ── Shutdown cleanup (optional) ─────────────────────────────
    logger.info("🛑 Server shutting down")


# -------------------------------------------------
# FastAPI app initialization (with lifespan)
# -------------------------------------------------
app = FastAPI(
    title="Weather Safety Service API",
    description="UV & weather-based risk prediction service for tourists",
    version="1.0.0",
    lifespan=lifespan,
)

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
# Fitzpatrick Skin Type Prediction Endpoint
# -------------------------------------------------
@app.post("/api/skin/fitzpatrick_predict")
async def predict_fitzpatrick(file: UploadFile = File(...)):
    if fitzpatrick_model is None:
        raise HTTPException(status_code=503, detail="Fitzpatrick model not loaded")

    image_pil = Image.open(file.file).convert("RGB")
    image_np = np.array(image_pil)

    from utils.validator import validate_human_skin
    validate_human_skin(image_np)

    image_tensor = preprocess_image(image_pil).to(device)

    with torch.no_grad():
        outputs = fitzpatrick_model(image_tensor)
        pred_class = outputs.argmax(dim=1).item() + 1

    return {"predicted_skin_type": pred_class}


# -------------------------------------------------
# Predict single weather risk instance
# -------------------------------------------------
@app.post("/api/weather/predict")
def predict_weather_risk(request: PredictRequest):
    if weather_service is None:
        raise HTTPException(status_code=503, detail="Weather risk model not loaded")

    try:
        service = get_weather_service()

        if service is None:
            return {"success": False, "error": "Weather service unavailable"}

        result = service.predict_one(request.features.model_dump())

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
    if weather_service is None:
        raise HTTPException(status_code=503, detail="Weather risk model not loaded")

    try:
        items = [f.dict() for f in request.features_list]
        results = weather_service.predict_batch(items)
        return {
            "success": True,
            "predictions": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/api/weather/health")
def health_check():
    status = weather_service.status() if weather_service else {}
    model_path = os.getenv("WEATHER_MODEL_FILE", "models/uv_risk_gradient_boosting.pkl")

    return {
        "status": "healthy",
        "model_loaded": status.get("model") == "loaded",
        "scaler_loaded": status.get("scaler") == "loaded",
        "model_file_exists": os.path.exists(model_path),
        "feature_count": status.get("feature_count"),
        "targets": ["low", "moderate", "high", "very high"]
    }


# -------------------------------------------------
# Run locally:  python app.py
# or:           uvicorn app:app --reload --port 8004
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
