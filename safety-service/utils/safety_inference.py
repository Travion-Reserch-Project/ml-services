import os
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'lat', 'lon', 'area_cluster', 'is_beach',
    'is_crowded', 'is_tourist_place', 'is_transit',
    'hour', 'day_of_week', 'is_weekend', 'police_nearby'
]

RISK_TARGETS = [
    'risk_harassment',
    'risk_pickpocket',
    'risk_scam',
    'risk_bag_snatching',
    'risk_theft',
    'risk_extortion',
    'risk_money_theft'
]

DEFAULT_LABELS = ["Low", "Medium", "High"]


class SafetyService:
    """
    Main service class for safety risk predictions.
    Mirrors transport-service structure.
    """
    
    def __init__(self, artifact_path: str):
        self.model = None
        self.scaler = None
        self.target_encoders = None
        self.meta: Dict[str, Any] = {
            "feature_cols": FEATURE_COLS,
            "targets": RISK_TARGETS,
            "labels": DEFAULT_LABELS
        }
        self._load_artifacts(artifact_path)

    def _load_artifacts(self, artifact_path: str):
        """Load model artifacts from disk or MLflow"""
        try:
            artifact = Path(artifact_path)
            if artifact.exists():
                # Primary: single pickle with dict {model, scaler, meta}
                obj = joblib.load(artifact)
                
                if isinstance(obj, dict):
                    self.model = obj.get("model")
                    self.scaler = obj.get("scaler")
                    self.target_encoders = obj.get("target_encoders")
                    meta = obj.get("meta")
                    if isinstance(meta, dict):
                        self.meta.update(meta)
                    logger.info("✓ Loaded unified safety artifact")
                else:
                    # Assume it's the model only; try to load companion files
                    self.model = obj
                    cache_dir = artifact.parent
                    scaler_fp = cache_dir / "safety_scaler.pkl"
                    meta_fp = cache_dir / "safety_meta.json"
                    if scaler_fp.exists():
                        self.scaler = joblib.load(scaler_fp)
                        logger.info("✓ Loaded safety scaler")
                    if meta_fp.exists():
                        with open(meta_fp, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            if isinstance(meta, dict):
                                self.meta.update(meta)
                        logger.info("✓ Loaded safety metadata")
                    logger.info("✓ Loaded safety model")
            else:
                logger.warning(f"⚠️ Safety model artifact not found at {artifact}")
        except Exception as e:
            logger.error(f"✗ Failed to load safety artifacts: {e}")

    def status(self) -> Dict[str, Any]:
        """Return service status"""
        return {
            "model": "loaded" if self.model is not None else "not_loaded",
            "scaler": "loaded" if self.scaler is not None else "not_loaded",
            "feature_count": len(self.meta.get("feature_cols", FEATURE_COLS)),
            "targets": self.meta.get("targets", RISK_TARGETS),
        }

    def _to_dataframe(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert input to DataFrame with proper column order"""
        df = pd.DataFrame(items)
        # Reorder/ensure all features present
        for col in self.meta.get("feature_cols", FEATURE_COLS):
            if col not in df.columns:
                df[col] = 0
        return df[self.meta.get("feature_cols", FEATURE_COLS)]

    def _preprocess(self, X_df: pd.DataFrame) -> np.ndarray:
        """Apply scaling if available"""
        X = X_df.values
        if self.scaler is not None:
            try:
                return self.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}, using raw features")
                return X
        return X

    def _postprocess(self, preds: np.ndarray) -> List[Dict[str, Any]]:
        """Convert model predictions to labeled format"""
        results: List[Dict[str, Any]] = []
        for prediction in preds:
            decoded_output: Dict[str, Any] = {}
            targets = self.meta.get("targets", RISK_TARGETS)
            
            for i, col in enumerate(targets):
                decoded_output[col] = self.target_encoders[col].inverse_transform(
                    [prediction[i]]
                )[0]

            results.append(decoded_output)

        return results

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict risk levels for a single location"""
        batch = self.predict_batch([features])
        return batch[0] if batch else {}

    def predict_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict risk levels for multiple locations"""
        X_df = self._to_dataframe(items)

        Xp = self._preprocess(X_df)

        if self.model is None:
            # Limited mode: heuristic baseline (very simple demo)
            # Higher risk at crowded tourist beaches at night without police nearby
            logger.warning("⚠️ Model not loaded, using heuristic baseline")
            results = []
            for i in range(len(X_df)):
                row = X_df.iloc[i]
                base = 0
                base += int(row.get('is_crowded', 0))
                base += int(row.get('is_tourist_place', 0))
                base += int(row.get('is_beach', 0))
                # Night hours 20:00-04:00
                hour = int(row.get('hour', 12))
                if hour >= 20 or hour <= 4:
                    base += 1
                if int(row.get('police_nearby', 0)) == 0:
                    base += 1
                # Map to Low/Medium/High
                if base <= 1:
                    label = "Low"
                elif base == 2:
                    label = "Medium"
                else:
                    label = "High"
                out = {t: label for t in self.meta.get("targets", RISK_TARGETS)}
                results.append(out)
            return results

        # Trained model path
        try:
            y_pred = self.model.predict(Xp)
            # If model returns DataFrame or list of arrays
            if isinstance(y_pred, list):
                y_pred = np.column_stack(y_pred)
            elif isinstance(y_pred, pd.DataFrame):
                y_pred = y_pred.values
            return self._postprocess(y_pred)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = SafetyService(
        artifact_path='model/safety_risk_model.pkl'
    )
    
    # Test prediction
    test_features = {
        'lat': 6.9271,
        'lon': 79.8612,
        'area_cluster': 0,
        'is_beach': 1,
        'is_crowded': 1,
        'is_tourist_place': 1,
        'is_transit': 0,
        'hour': 22,
        'day_of_week': 5,
        'is_weekend': 1,
        'police_nearby': 0
    }
    
    print("\n🧪 Testing Safety Service:\n")
    result = service.predict_one(test_features)
    print(f"Risk Predictions for location ({test_features['lat']}, {test_features['lon']}):")
    for risk_type, level in result.items():
        print(f"   {risk_type}: {level}")
