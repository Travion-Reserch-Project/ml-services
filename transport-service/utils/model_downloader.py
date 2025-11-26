"""
Model Downloader for ML Services
Downloads models from GitHub Releases, S3, GCS, or Azure Blob Storage
"""
import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                "gnn_model": {
                    "source": "github_release",
                    "repo": "Travion-Reserch-Project/ml-services",
                    "tag": "latest",
                    "filename": "transport_gnn_model.pth",
                    "hash": "abc123..."
                },
                "risk_model": {
                    "source": "url",
                    "url": "https://example.com/risk_model.pkl",
                    "filename": "risk_model.pkl"
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


def download_transport_models() -> Dict[str, Path]:
    """
    Download all transport service models
    
    Returns:
        Dictionary of model paths
    """
    downloader = ModelDownloader(cache_dir="model")
    
    # Configuration from environment variables
    repo = os.getenv("MODEL_REPO", "Travion-Reserch-Project/ml-services")
    release_tag = os.getenv("MODEL_RELEASE_TAG", "latest")
    
    model_config = {
        "gnn_model": {
            "source": "github_release",
            "repo": repo,
            "tag": release_tag,
            "filename": "transport_gnn_model.pth",
            # Hash will be added when model is uploaded
        },
        "risk_model": {
            "source": "github_release",
            "repo": repo,
            "tag": release_tag,
            "filename": "risk_model.pkl",
        }
    }
    
    logger.info("=" * 60)
    logger.info("Starting model download...")
    logger.info(f"Repository: {repo}")
    logger.info(f"Release tag: {release_tag}")
    logger.info("=" * 60)
    
    models = downloader.download_models(model_config)
    
    logger.info("=" * 60)
    logger.info(f"✓ Downloaded {len(models)}/{len(model_config)} models")
    logger.info("=" * 60)
    
    return models


if __name__ == "__main__":
    # Test the downloader
    download_transport_models()
