from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

router = APIRouter(tags=["safety-routes"])

# Mirror schemas to keep routes modular
class SafetyFeatures(BaseModel):
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

class BatchPredictRequest(BaseModel):
    features_list: List[SafetyFeatures]


@router.post("/routes/batch_predict")
def routes_batch_predict(req: BatchPredictRequest):
    """
    Additional batch predict route (modular pattern for refactoring).
    This demonstrates how to add routes in separate files.
    """
    try:
        # Delegate to core app service to avoid duplicate loaders
        from ..app import get_safety_service, ensure_models_downloaded
        ensure_models_downloaded()
        service = get_safety_service()
        items = [f.dict() for f in req.features_list]
        result = service.predict_batch(items)
        return {"success": True, "predictions": result, "count": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
