# Standard library imports
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# -------------------------------------------------
# Logging configuration
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Default feature columns (must match training)
# -------------------------------------------------
DEFAULT_FEATURE_COLS = [
    'Skin Type_encoded',
    'UV Index',
    'Time of Day',
    'Historical Sunburn',
    'Historical Tanning',
    'Skin Product Interaction_encoded',
    'Use of Sunglasses/Hat/Shade_encoded',
    'Cloud Cover_encoded',
    'Age',
    'Temperature_C',
    'Humidity_%'
]

PROTECTION_MAPPING = {
    "never": 0,
    "rarely": 1,
    "sometimes": 2,
    "often": 3,
    "always": 4
}

CLOUD_COVER_MAPPING = {
    "clear": 0,
    "partly cloudy": 1,
    "overcast": 2
}

# Final human-readable targets
RISK_TARGETS = ["low", "moderate", "high", "very high"]


class WeatherRiskService:
    """
    Service class for UV & weather-based risk prediction.

    Handles:
    - Loading model + metadata
    - Input preprocessing
    - Prediction (single & batch)
    - Output decoding
    """

    def __init__(self, model_path: str):
        self.model = None
        self.scaler = None
        self.feature_cols = DEFAULT_FEATURE_COLS
        self.decode_map: Dict[int, str] = {}

        self._load_artifacts(model_path)

    # -------------------------------------------------
    # Load model artifacts
    # -------------------------------------------------
    def _load_artifacts(self, model_path: str):
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                logger.warning(f"⚠️ Weather model not found at {model_path}")
                return

            # Load trained model
            self.model = joblib.load(model_path)
            logger.info("✓ Loaded weather risk model")

            # Load feature list if exists
            features_fp = model_path.parent / "model_features.pkl"
            if features_fp.exists():
                self.feature_cols = joblib.load(features_fp)
                logger.info("✓ Loaded model feature list")

            # Load decode map
            decode_fp = model_path.parent / "risk_decode_map.pkl"
            if decode_fp.exists():
                self.decode_map = joblib.load(decode_fp)
                logger.info("✓ Loaded risk decode map")
            else:
                # Fallback mapping (safe default)
                self.decode_map = {
                    0: "low",
                    1: "moderate",
                    2: "high",
                    3: "very high"
                }
                logger.warning("⚠️ Using default risk decode map")

        except Exception as e:
            logger.error(f"✗ Failed to load weather model artifacts: {e}")

    # -------------------------------------------------
    # Service status (used by /health)
    # -------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "model": "loaded" if self.model is not None else "not_loaded",
            "scaler": "not_loaded",  # No scaler in your pipeline
            "feature_count": len(self.feature_cols),
            "targets": RISK_TARGETS
        }

    # -------------------------------------------------
    # Convert input to DataFrame
    # -------------------------------------------------
    def _to_dataframe(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        
        encoded_items = [self._preprocess(item) for item in items]
        df = pd.DataFrame(encoded_items)

        # Ensure all required features exist
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_cols]

    # -------------------------------------------------
    # Preprocess
    # -------------------------------------------------
    def _preprocess(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode raw user input into model-ready numerical features.
        MUST match training feature columns exactly.
        """

        return {
            # LabelEncoder behaviour for Skin Type (1–6 → 0–5)
            "Skin Type_encoded": raw["Skin_Type"] - 1,

            # Numerical features
            "UV Index": raw["UV_Index"],
            "Time of Day": raw["Time_of_Day"],
            "Historical Sunburn": raw["Historical_Sunburn"],
            "Historical Tanning": raw["Historical_Tanning"],
            "Age": raw["Age"],
            "Temperature_C": raw["Temperature_C"],
            "Humidity_%": raw["Humidity_pct"],

            # Ordinal encodings
            "Skin Product Interaction_encoded": PROTECTION_MAPPING.get(
                raw["Skin_Product_Interaction"].lower(), 0
            ),
            "Use of Sunglasses/Hat/Shade_encoded": PROTECTION_MAPPING.get(
                raw["Use_of_Sunglasses"].lower(), 0
            ),

            # LabelEncoder behaviour for Cloud Cover
            "Cloud Cover_encoded": CLOUD_COVER_MAPPING.get(
                raw["Cloud_Cover"].lower(), 0
            )
        }

    # -------------------------------------------------
    # Postprocess predictions
    # -------------------------------------------------
    def _postprocess(self, preds: np.ndarray) -> List[Dict[str, Any]]:
        results = []

        for p in preds:
            # Ensure scalar
            encoded = int(p)
            label = self.decode_map.get(encoded, "low")

            results.append({
                "risk_level": label,
                "encoded_value": encoded
            })

        return results

    # -------------------------------------------------
    # Predict single instance
    # -------------------------------------------------
    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        batch = self.predict_batch([features])
        return batch[0] if batch else {}

    # -------------------------------------------------
    # Predict batch
    # -------------------------------------------------
    def predict_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        X_df = self._to_dataframe(items)

        # -------------------------------------------------
        # Fallback heuristic if model missing
        # -------------------------------------------------
        if self.model is None:
            logger.warning("⚠️ Model not loaded – using heuristic fallback")
            results = []

            for _, row in X_df.iterrows():
                score = 0

                if row["uv_index"] >= 8:
                    score += 2
                elif row["uv_index"] >= 5:
                    score += 1

                if 11 <= row["time_of_day"] <= 14:
                    score += 1

                if row["temperature_c"] >= 32:
                    score += 1

                if row["humidity_pct"] <= 60:
                    score += 1

                if row["skin_type"] <= 2:
                    score += 1

                if score <= 1:
                    label = "low"
                elif score == 2:
                    label = "moderate"
                elif score == 3:
                    label = "high"
                else:
                    label = "very high"

                results.append({"risk_level": label})

            return results

        # -------------------------------------------------
        # Normal ML prediction
        # -------------------------------------------------
        try:
            y_pred = self.model.predict(X_df.values)

            if isinstance(y_pred, list):
                y_pred = np.array(y_pred)

            return self._postprocess(y_pred)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# -------------------------------------------------
# Local testing
# -------------------------------------------------
if __name__ == "__main__":
    service = WeatherRiskService(
        model_path="models/uv_risk_gradient_boosting.pkl"
    )

    test_input = {
        "skin_type": 2,
        "uv_index": 9,
        "time_of_day": 13,
        "historical_sunburn": 1,
        "historical_tanning": 0,
        "age": 28,
        "temperature_c": 34,
        "humidity_pct": 55
    }

    print("\n🧪 Testing WeatherRiskService:\n")
    print(service.predict_one(test_input))
