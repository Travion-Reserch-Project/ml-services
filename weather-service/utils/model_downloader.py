"""
Model Downloader for Weather / UV Risk Service

Supports:
- MLflow Model Registry (DagsHub)
- GitHub Releases (fallback)

Responsible for downloading, validating, and preparing
UV risk model artifacts for inference.
"""

import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Optional MLflow import
# -------------------------------------------------
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    logger.info("✓ MLflow available")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠️ MLflow not installed – MLflow loading disabled")


# =================================================
# MLflow Loader (Weather model)
# =================================================
class MLflowModelLoader:
    """Load weather models from MLflow Model Registry (DagsHub)"""

    def __init__(self, tracking_uri: Optional[str] = None):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not installed")

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"📊 MLflow tracking URI: {tracking_uri}")

    def load_model(
        self,
        model_name: str,
        stage: str = "Production",
        version: Optional[str] = None
    ):
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
                logger.info(f"📦 Loading {model_name} v{version}")
            else:
                model_uri = f"models:/{model_name}/{stage}"
                logger.info(f"📦 Loading {model_name} ({stage})")

            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Loaded model from MLflow: {model_uri}")
            return model

        except Exception as e:
            logger.error(f"❌ MLflow load failed: {e}")
            raise


# =================================================
# Generic Downloader
# =================================================
class ModelDownloader:
    """Download and cache model artifacts"""

    def __init__(self, cache_dir: str = "models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_hash(self, filepath: Path) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _is_cached(self, filename: str, expected_hash: Optional[str] = None) -> bool:
        filepath = self.cache_dir / filename

        if not filepath.exists():
            return False

        if expected_hash:
            if self._get_file_hash(filepath) != expected_hash:
                logger.warning(f"Hash mismatch for {filename}")
                return False

        logger.info(f"✓ Cached: {filename}")
        return True

    def download_from_url(
        self,
        url: str,
        filename: str,
        expected_hash: Optional[str] = None
    ) -> Path:
        filepath = self.cache_dir / filename

        if self._is_cached(filename, expected_hash):
            return filepath

        logger.info(f"⬇️ Downloading {filename}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if expected_hash:
            if self._get_file_hash(filepath) != expected_hash:
                filepath.unlink()
                raise ValueError("Hash verification failed")

        logger.info(f"✓ Downloaded {filename}")
        return filepath

    def download_from_github_release(
        self,
        repo: str,
        tag: str,
        filename: str,
        expected_hash: Optional[str] = None
    ) -> Path:
        if tag == "latest":
            url = f"https://github.com/{repo}/releases/latest/download/{filename}"
        else:
            url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"

        return self.download_from_url(url, filename, expected_hash)

    def download_models(self, model_config: Dict[str, Dict]) -> Dict[str, Path]:
        downloaded = {}

        for name, cfg in model_config.items():
            try:
                filepath = self.download_from_github_release(
                    repo=cfg["repo"],
                    tag=cfg.get("tag", "latest"),
                    filename=cfg["filename"],
                    expected_hash=cfg.get("hash")
                )
                downloaded[name] = filepath
            except Exception as e:
                logger.error(f"✗ Failed to download {name}: {e}")

        return downloaded


# =================================================
# WEATHER MODEL DOWNLOADER (MAIN ENTRY)
# =================================================
def download_weather_models() -> Dict[str, Any]:
    """
    Download all weather service model artifacts.

    Returns:
        Dictionary of downloaded artifact paths
    """

    model_source = os.getenv("MODEL_SOURCE", "github_release")

    logger.info("=" * 60)
    logger.info(f"Model Source: {model_source}")
    logger.info("=" * 60)

    # -------------------------------
    # Option 1: MLflow (DagsHub)
    # -------------------------------
    if model_source == "mlflow" and MLFLOW_AVAILABLE:
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            model_name = os.getenv("MODEL_NAME", "uv-risk-model")
            stage = os.getenv("MODEL_STAGE", "Production")

            if not tracking_uri:
                raise ValueError("MLFLOW_TRACKING_URI not set")

            # Credentials
            if os.getenv("MLFLOW_TRACKING_USERNAME"):
                os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
                os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

            loader = MLflowModelLoader(tracking_uri)
            model = loader.load_model(model_name, stage=stage)

            os.makedirs("models", exist_ok=True)
            model_path = "models/uv_risk_model.pkl"
            joblib.dump(model, model_path)

            logger.info("✅ Weather model loaded from MLflow")

            return {
                "weather_model": model_path,
                "source": "mlflow",
                "stage": stage
            }

        except Exception as e:
            logger.error(f"❌ MLflow loading failed: {e}")
            logger.info("💡 Falling back to GitHub Releases")

    # -------------------------------
    # Option 2: GitHub Releases
    # -------------------------------
    downloader = ModelDownloader(cache_dir="models")

    repo = os.getenv("MODEL_REPO", "Travion-Reserch-Project/ml-services")
    tag = os.getenv("MODEL_RELEASE_TAG", "latest")

    model_config = {
        "weather_model": {
            "repo": repo,
            "tag": tag,
            "filename": "uv_risk_model.pkl"
        },
        "features": {
            "repo": repo,
            "tag": tag,
            "filename": "model_features.pkl"
        },
        "decode_map": {
            "repo": repo,
            "tag": tag,
            "filename": "risk_decode_map.pkl"
        }
    }

    logger.info("📦 Downloading weather model artifacts from GitHub Releases")
    models = downloader.download_models(model_config)

    logger.info("=" * 60)
    logger.info(f"✓ Downloaded {len(models)}/{len(model_config)} artifacts")
    logger.info("=" * 60)

    return models


def download_weather_models() -> Dict[str, Any]:
    """
    Download all weather service models.
    Supports MLflow (DagsHub) or GitHub Releases fallback.

    Returns:
        Dictionary of downloaded model artifact paths
    """
    model_source = os.getenv("MODEL_SOURCE", "github_release")

    logger.info("=" * 60)
    logger.info(f"Model Source: {model_source}")
    logger.info("=" * 60)

    # =====================================================
    # OPTION 1: MLflow Model Registry (DagsHub)
    # =====================================================
    if model_source == "mlflow":
        if not MLFLOW_AVAILABLE:
            logger.error("❌ MLflow not installed but MODEL_SOURCE=mlflow")
            logger.info("💡 Falling back to github_release...")
            model_source = "github_release"
        else:
            try:
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                model_name = os.getenv("MODEL_NAME", "uv-risk-model")
                model_stage = os.getenv("MODEL_STAGE", "Production")
                model_version = os.getenv("MODEL_VERSION")  # optional

                if not tracking_uri:
                    raise ValueError("MLFLOW_TRACKING_URI is required")

                # Credentials
                username = os.getenv("MLFLOW_TRACKING_USERNAME")
                password = os.getenv("MLFLOW_TRACKING_PASSWORD")
                if username and password:
                    os.environ["MLFLOW_TRACKING_USERNAME"] = username
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

                loader = MLflowModelLoader(tracking_uri)

                logger.info("📊 Loading weather model from MLflow...")
                logger.info(f"   Model: {model_name}")
                logger.info(f"   Stage: {model_stage}")

                model = loader.load_model(
                    model_name=model_name,
                    stage=model_stage if not model_version else None,
                    version=model_version
                )

                # Save locally for WeatherRiskService
                os.makedirs("models", exist_ok=True)
                model_path = "models/uv_risk_model.pkl"
                joblib.dump(model, model_path)

                logger.info("✅ Model loaded from MLflow and saved locally")

                return {
                    "weather_model": model_path,
                    "source": "mlflow",
                    "model_name": model_name,
                    "stage": model_stage
                }

            except Exception as e:
                logger.error(f"❌ MLflow loading failed: {e}")
                logger.info("💡 Falling back to github_release...")
                model_source = "github_release"

    # =====================================================
    # OPTION 2: GitHub Releases (Fallback)
    # =====================================================
    if model_source == "github_release":
        downloader = ModelDownloader(cache_dir="models")

        repo = os.getenv("MODEL_REPO", "Travion-Reserch-Project/ml-services")
        release_tag = os.getenv("MODEL_RELEASE_TAG", "latest")

        model_config = {
            "weather_model": {
                "repo": repo,
                "tag": release_tag,
                "filename": "uv_risk_model.pkl",
            },
            "features": {
                "repo": repo,
                "tag": release_tag,
                "filename": "model_features.pkl",
            },
            "decode_map": {
                "repo": repo,
                "tag": release_tag,
                "filename": "risk_decode_map.pkl",
            }
        }

        logger.info("📦 Downloading weather models from GitHub Releases...")
        logger.info(f"   Repository: {repo}")
        logger.info(f"   Tag: {release_tag}")

        models = downloader.download_models(model_config)

        logger.info("=" * 60)
        logger.info(f"✓ Downloaded {len(models)}/{len(model_config)} artifacts")
        logger.info("=" * 60)

        return models

    logger.info("=" * 60)
    return {}


if __name__ == "__main__":
    # Test the downloader
    download_safety_models()