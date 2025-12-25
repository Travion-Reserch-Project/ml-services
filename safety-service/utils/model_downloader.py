"""
Model Downloader for Safety Service
Supports: MLflow Model Registry (DagsHub), DVC, GitHub Releases, S3, GCS, Azure
Mirrors transport-service downloader patterns for consistency
"""
import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional MLflow import (graceful fallback)
try:
    import mlflow
    import mlflow.pyfunc
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    logger.info("✓ MLflow available")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("⚠️  MLflow not installed - MLflow features disabled")


class MLflowModelLoader:
    """Load models from MLflow Model Registry (DagsHub)"""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow model loader
        
        Args:
            tracking_uri: MLflow tracking URI (e.g., https://dagshub.com/user/repo.mlflow)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not installed. Run: pip install mlflow")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"📊 MLflow tracking URI: {tracking_uri}")
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = "Production"
    ) -> Any:
        """
        Load model from MLflow Model Registry
        
        Args:
            model_name: Registered model name (e.g., "safety-risk-model")
            version: Specific version (e.g., "3") or None
            stage: Stage name ("Production", "Staging", "None") or None
            
        Returns:
            Loaded model object
            
        Example:
            loader = MLflowModelLoader("https://dagshub.com/user/repo.mlflow")
            model = loader.load_model("safety-risk", stage="Production")
        """
        try:
            # Build model URI
            if version:
                model_uri = f"models:/{model_name}/{version}"
                logger.info(f"📦 Loading {model_name} v{version}...")
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
                logger.info(f"📦 Loading {model_name} from {stage}...")
            else:
                # Get latest version
                client = mlflow.MlflowClient()
                versions = client.search_model_versions(f"name='{model_name}'")
                if not versions:
                    raise ValueError(f"No versions found for model '{model_name}'")
                latest_version = max(versions, key=lambda v: int(v.version))
                model_uri = f"models:/{model_name}/{latest_version.version}"
                logger.info(f"📦 Loading {model_name} latest v{latest_version.version}...")
            
            # Load model
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"✅ Model loaded: {model_uri}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load from MLflow: {e}")
            raise
    
    def get_model_info(self, model_name: str, stage: str = "Production") -> Dict:
        """Get model metadata from registry"""
        try:
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            
            if not versions:
                return {"error": f"No {stage} version found for {model_name}"}
            
            version = versions[0]
            return {
                "name": model_name,
                "version": version.version,
                "stage": stage,
                "run_id": version.run_id,
                "status": version.status,
                "created_at": version.creation_timestamp
            }
        except Exception as e:
            return {"error": str(e)}


class ModelDownloader:
    """Download and cache ML models from various sources"""
    
    def __init__(self, cache_dir: str = "model"):
        """
        Initialize model downloader
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def _get_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _is_cached(self, filename: str, expected_hash: Optional[str] = None) -> bool:
        """Check if model is already cached and valid"""
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            return False
        
        # If hash provided, verify integrity
        if expected_hash:
            actual_hash = self._get_file_hash(filepath)
            if actual_hash != expected_hash:
                logger.warning(f"Hash mismatch for {filename}. Re-downloading...")
                return False
        
        logger.info(f"✓ Model {filename} found in cache")
        return True
    
    def download_from_url(
        self, 
        url: str, 
        filename: str, 
        expected_hash: Optional[str] = None,
        force: bool = False
    ) -> Path:
        """
        Download model from URL
        
        Args:
            url: Download URL
            filename: Local filename to save as
            expected_hash: Optional SHA256 hash for verification
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded model file
        """
        filepath = self.cache_dir / filename
        
        # Check cache
        if not force and self._is_cached(filename, expected_hash):
            return filepath
        
        logger.info(f"⬇️  Downloading {filename} from {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"   {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
            
            logger.info(f"✓ Downloaded {filename} successfully")
            
            # Verify hash if provided
            if expected_hash:
                actual_hash = self._get_file_hash(filepath)
                if actual_hash != expected_hash:
                    filepath.unlink()  # Delete corrupted file
                    raise ValueError(f"Hash verification failed for {filename}")
                logger.info(f"✓ Hash verified for {filename}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"✗ Failed to download {filename}: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def download_from_github_release(
        self,
        repo: str,
        tag: str,
        filename: str,
        expected_hash: Optional[str] = None,
        token: Optional[str] = None
    ) -> Path:
        """
        Download model from GitHub Release
        
        Args:
            repo: Repository in format 'owner/repo'
            tag: Release tag (e.g., 'v1.0.0' or 'latest')
            filename: Asset filename to download
            expected_hash: Optional SHA256 hash
            token: Optional GitHub token for private repos
            
        Returns:
            Path to downloaded model file
        """
        # Build GitHub Release URL
        if tag == "latest":
            url = f"https://github.com/{repo}/releases/latest/download/{filename}"
        else:
            url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
        
        logger.info(f"📦 Downloading from GitHub Release: {repo}@{tag}")
        
        # Add auth header if token provided
        if token:
            response = requests.get(url, stream=True, timeout=300, 
                                   headers={"Authorization": f"token {token}"})
            response.raise_for_status()
            return self._download_from_response(response, filename, expected_hash)
        else:
            return self.download_from_url(url, filename, expected_hash)
    
    def _download_from_response(self, response, filename: str, expected_hash: Optional[str]) -> Path:
        """Helper to download from requests response"""
        filepath = self.cache_dir / filename
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        if expected_hash:
            actual_hash = self._get_file_hash(filepath)
            if actual_hash != expected_hash:
                filepath.unlink()
                raise ValueError(f"Hash verification failed for {filename}")
        
        return filepath
    
    def download_models(self, model_config: Dict) -> Dict[str, Path]:
        """
        Download multiple models from configuration
        
        Args:
            model_config: Dictionary with model configurations
            
        Example:
            {
                "risk_model": {
                    "source": "github_release",
                    "repo": "Travion-Reserch-Project/ml-services",
                    "tag": "latest",
                    "filename": "safety_risk_model.pkl",
                    "hash": "abc123..."
                },
                "scaler": {
                    "source": "url",
                    "url": "https://example.com/safety_scaler.pkl",
                    "filename": "safety_scaler.pkl"
                }
            }
            
        Returns:
            Dictionary mapping model names to local file paths
        """
        downloaded_models = {}
        
        for model_name, config in model_config.items():
            try:
                source = config.get("source", "url")
                filename = config["filename"]
                expected_hash = config.get("hash")
                
                if source == "github_release":
                    filepath = self.download_from_github_release(
                        repo=config["repo"],
                        tag=config.get("tag", "latest"),
                        filename=filename,
                        expected_hash=expected_hash,
                        token=os.getenv("GITHUB_TOKEN")
                    )
                elif source == "url":
                    filepath = self.download_from_url(
                        url=config["url"],
                        filename=filename,
                        expected_hash=expected_hash
                    )
                else:
                    logger.warning(f"Unknown source type '{source}' for {model_name}")
                    continue
                
                downloaded_models[model_name] = filepath
                logger.info(f"✓ {model_name}: {filepath}")
                
            except Exception as e:
                logger.error(f"✗ Failed to download {model_name}: {e}")
                # Continue with other models
                continue
        
        return downloaded_models


def download_safety_models() -> Dict[str, Any]:
    """
    Download all safety service models
    Supports MLflow (DagsHub), DVC, or GitHub Releases
    
    Returns:
        Dictionary of model paths or loaded models
    """
    # Check model source from environment
    model_source = os.getenv("MODEL_SOURCE", "github_release")
    
    logger.info("=" * 60)
    logger.info(f"Model Source: {model_source}")
    logger.info("=" * 60)
    
    # Option 1: Load from MLflow Model Registry (DagsHub)
    if model_source == "mlflow":
        if not MLFLOW_AVAILABLE:
            logger.error("❌ MLflow not installed but MODEL_SOURCE=mlflow")
            logger.info("💡 Falling back to github_release...")
            model_source = "github_release"
        else:
            try:
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                model_name = os.getenv("MODEL_NAME", "safety-risk-model")
                model_stage = os.getenv("MODEL_STAGE", "Production")
                model_version = os.getenv("MODEL_VERSION")  # Optional
                
                if not tracking_uri:
                    logger.error("❌ MLFLOW_TRACKING_URI not set")
                    raise ValueError("MLflow URI required when MODEL_SOURCE=mlflow")
                
                # Set MLflow credentials
                username = os.getenv("MLFLOW_TRACKING_USERNAME")
                password = os.getenv("MLFLOW_TRACKING_PASSWORD")
                if username and password:
                    os.environ["MLFLOW_TRACKING_USERNAME"] = username
                    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
                
                loader = MLflowModelLoader(tracking_uri)
                
                logger.info(f"📊 Loading from MLflow Registry...")
                logger.info(f"   Model: {model_name}")
                logger.info(f"   Stage: {model_stage}")
                if model_version:
                    logger.info(f"   Version: {model_version}")
                
                model = loader.load_model(
                    model_name=model_name,
                    version=model_version,
                    stage=model_stage if not model_version else None
                )
                
                logger.info("✅ Model loaded from MLflow")
                return {
                    "safety_model": model,
                    "source": "mlflow",
                    "model_name": model_name,
                    "stage": model_stage
                }
                
            except Exception as e:
                logger.error(f"❌ MLflow loading failed: {e}")
                logger.info("💡 Falling back to github_release...")
                model_source = "github_release"
    
    # Option 2: GitHub Releases (Fallback)
    if model_source == "github_release":
        downloader = ModelDownloader(cache_dir="model")
        
        repo = os.getenv("MODEL_REPO", "Travion-Reserch-Project/ml-services")
        release_tag = os.getenv("MODEL_RELEASE_TAG", "latest")
        
        model_config = {
            "safety_model": {
                "source": "github_release",
                "repo": repo,
                "tag": release_tag,
                "filename": "safety_risk_model.pkl",
            },
            "scaler": {
                "source": "github_release",
                "repo": repo,
                "tag": release_tag,
                "filename": "safety_scaler.pkl",
            },
            "meta": {
                "source": "github_release",
                "repo": repo,
                "tag": release_tag,
                "filename": "safety_meta.json",
            }
        }
        
        logger.info(f"📦 Downloading from GitHub Releases...")
        logger.info(f"   Repository: {repo}")
        logger.info(f"   Tag: {release_tag}")
        
        models = downloader.download_models(model_config)
        
        logger.info("=" * 60)
        logger.info(f"✓ Downloaded {len(models)}/{len(model_config)} models")
        logger.info("=" * 60)
        
        return models
    
    logger.info("=" * 60)
    return {}


if __name__ == "__main__":
    # Test the downloader
    download_safety_models()
