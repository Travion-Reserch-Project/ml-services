#!/usr/bin/env python3
"""
Download trained model from DagsHub MLflow artifact store
Industry standard approach: Separate model training from model serving

Enhancements:
- Respect MLFLOW_TRACKING_URI/USERNAME/PASSWORD if provided
- Fallback to DAGSHUB_REPO_OWNER/NAME + DAGSHUB_TOKEN
- Recursively search run artifacts to find a .pth file
- Handle Git LFS pointers (DagsHub stores large files with LFS)
"""
import os
import sys
import re
import requests
from pathlib import Path
import mlflow

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Also try parent directory .env
    parent_env = Path(__file__).parent.parent / '.env'
    if parent_env.exists():
        load_dotenv(parent_env)
except ImportError:
    pass  # dotenv not available, rely on system env vars


def _find_pth_artifact_recursive(client: mlflow.tracking.MlflowClient, run_id: str, path: str = "") -> str:
    """Recursively search artifacts for a .pth file and return its artifact path.
    
    Priority: Look for 'model/transport_gnn_model.pth' first (complete bundle),
    otherwise find any .pth file.
    """
    items = client.list_artifacts(run_id, path) if path else client.list_artifacts(run_id)
    
    # First pass: look for the canonical model bundle
    for it in items:
        if it.is_dir:
            found = _find_pth_artifact_recursive(client, run_id, it.path)
            if found:
                return found
        else:
            # Prefer the complete bundle
            if 'transport_gnn_model.pth' in it.path and it.path.endswith('.pth'):
                return it.path
    
    # Second pass: any .pth file as fallback
    for it in items:
        if not it.is_dir and it.path.endswith('.pth'):
            return it.path
    
    return ""


def _is_git_lfs_pointer(file_path: Path) -> tuple[bool, dict]:
    """Check if a file is a Git LFS pointer and extract metadata.
    
    Returns:
        (is_lfs, metadata_dict) where metadata contains 'oid' and 'size'
    """
    try:
        if file_path.stat().st_size > 500:  # LFS pointers are tiny
            return False, {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(500)
        
        # Git LFS pointer format:
        # version https://git-lfs.github.com/spec/v1
        # oid sha256:xxxxx
        # size 12345
        if 'git-lfs.github.com' not in content:
            return False, {}
        
        oid_match = re.search(r'oid sha256:([a-f0-9]{64})', content)
        size_match = re.search(r'size (\d+)', content)
        
        if oid_match and size_match:
            return True, {
                'oid': oid_match.group(1),
                'size': int(size_match.group(1))
            }
        return False, {}
    except Exception:
        return False, {}


def _download_lfs_file(lfs_metadata: dict, target_path: Path, dagshub_user: str, dagshub_repo: str, dagshub_token: str) -> bool:
    """Download the actual file from DagsHub LFS storage using the OID.
    
    Args:
        lfs_metadata: Dict with 'oid' and 'size' from LFS pointer
        target_path: Where to save the downloaded file
        dagshub_user: DagsHub username
        dagshub_repo: Repository name
        dagshub_token: Authentication token
    """
    oid = lfs_metadata['oid']
    expected_size = lfs_metadata['size']
    
    # Try multiple DagsHub LFS download patterns
    urls_to_try = [
        # Pattern 1: Direct storage with bearer auth
        f"https://dagshub.com/api/v1/repos/{dagshub_user}/{dagshub_repo}/storage/raw/{oid}",
        # Pattern 2: Git LFS media endpoint
        f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.git/info/lfs/objects/{oid}",
        # Pattern 3: DVC storage
        f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.dvc/raw/{oid}",
    ]
    
    for i, url in enumerate(urls_to_try, 1):
        try:
            print(f"🔍 Attempt {i}/{len(urls_to_try)}: {url[:80]}...")
            
            headers = {}
            auth = None
            
            if dagshub_token:
                if 'api/v1' in url:
                    # API endpoints use Bearer token
                    headers['Authorization'] = f'Bearer {dagshub_token}'
                else:
                    # Git endpoints use basic auth
                    auth = (dagshub_user, dagshub_token)
            
            response = requests.get(url, headers=headers, auth=auth, stream=True, timeout=300)
            
            if response.status_code == 404:
                print(f"   ❌ Not found, trying next pattern...")
                continue
            
            response.raise_for_status()
            
            print(f"   ✅ Found! Downloading ({expected_size / (1024*1024):.2f} MB)...")
            
            downloaded = 0
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (5 * 1024 * 1024) == 0:  # Log every 5 MB
                            progress = (downloaded / expected_size) * 100 if expected_size > 0 else 0
                            print(f"      {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
            
            actual_size = target_path.stat().st_size
            print(f"   ✅ Downloaded successfully ({actual_size / (1024*1024):.2f} MB)")
            
            # Size validation (allow some tolerance)
            if expected_size > 0 and abs(actual_size - expected_size) > 1024:
                print(f"   ⚠️  Size mismatch: expected {expected_size}, got {actual_size}")
            
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                continue  # Try next URL
            print(f"   ❌ HTTP error: {e}")
            continue
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"❌ All download attempts failed for OID {oid[:16]}")
    return False


def download_model_from_dagshub():
    """Download the latest model from DagsHub MLflow."""

    # Prefer explicit MLflow env vars if set
    explicit_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()

    # DagsHub convenience envs
    dagshub_user = os.getenv('DAGSHUB_REPO_OWNER', 'iamsahan').strip()
    dagshub_repo = os.getenv('DAGSHUB_REPO_NAME', 'ml-services').strip()
    dagshub_token = os.getenv('DAGSHUB_TOKEN', '').strip()

    # Derive tracking URI
    tracking_uri = explicit_tracking_uri or f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"

    # Credentials precedence: MLFLOW_* if present, else DAGSHUB_TOKEN + user
    mlflow_user = os.getenv('MLFLOW_TRACKING_USERNAME', '').strip() or dagshub_user
    mlflow_pass = os.getenv('MLFLOW_TRACKING_PASSWORD', '').strip() or dagshub_token

    if mlflow_user and mlflow_pass:
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_pass

    mlflow.set_tracking_uri(tracking_uri)

    print(f"🔍 Connecting to DagsHub MLflow: {tracking_uri}")
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Check if specific run ID is provided
        specific_run_id = os.getenv('MLFLOW_RUN_ID', '').strip()
        
        if specific_run_id:
            print(f"📦 Using specified run ID: {specific_run_id}")
            run_id = specific_run_id
        else:
            # Get the experiment
            experiment = mlflow.get_experiment_by_name('transport-gnn')
            
            if not experiment:
                # Try Default experiment
                experiment = mlflow.get_experiment_by_name('Default')
            
            if not experiment:
                print("⚠️  No suitable experiment found")
                return False
            
            # Get the latest run with the model artifact
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                print("⚠️  No runs found in experiment")
                return False
            
            latest_run = runs[0]
            run_id = latest_run.info.run_id
            
            print(f"📦 Found latest run: {run_id}")
        # Recursively search artifacts to find a .pth file
        model_artifact = _find_pth_artifact_recursive(client, run_id)

        if not model_artifact:
            print("⚠️  No .pth model file found in run artifacts")
            return False
        
        # Download the artifact
        model_dir = Path('model')
        model_dir.mkdir(exist_ok=True)
        
        print(f"⬇️  Downloading {model_artifact}...")
        
        artifact_path = client.download_artifacts(run_id, model_artifact, dst_path=str(model_dir))
        
        # Move to expected location if needed
        downloaded_file = Path(artifact_path)
        target_file = model_dir / 'transport_gnn_model.pth'
        
        if downloaded_file != target_file:
            downloaded_file.rename(target_file)
        
        # Check if it's a Git LFS pointer instead of actual file
        is_lfs, lfs_metadata = _is_git_lfs_pointer(target_file)
        if is_lfs:
            print("📌 Downloaded file is a Git LFS pointer, fetching actual binary...")
            
            # Get DagsHub credentials
            dagshub_user_lfs = os.getenv('DAGSHUB_REPO_OWNER', 'iamsahan').strip()
            dagshub_repo_lfs = os.getenv('DAGSHUB_REPO_NAME', 'ml-services').strip()
            dagshub_token_lfs = os.getenv('DAGSHUB_TOKEN', '').strip() or mlflow_pass
            
            if not _download_lfs_file(lfs_metadata, target_file, dagshub_user_lfs, dagshub_repo_lfs, dagshub_token_lfs):
                print("❌ Failed to download LFS file")
                return False
        
        # Validate downloaded file
        if not target_file.exists():
            print("❌ Download failed: file not found")
            return False
        
        file_size = target_file.stat().st_size
        if file_size < 10 * 1024:  # Less than 10KB is likely an error page/denied
            print(f"❌ Downloaded file too small ({file_size} bytes) - likely not a model")
            target_file.unlink()  # Delete corrupted file
            return False
        
        # Try to validate it's a valid PyTorch file
        try:
            import torch
            torch.load(str(target_file), map_location='cpu', weights_only=False)
            print(f"✅ Model validated successfully")
        except Exception as e:
            print(f"❌ Downloaded file is not a valid PyTorch model: {e}")
            target_file.unlink()  # Delete corrupted file
            return False
        
        print(f"✅ Model downloaded successfully to {target_file} ({file_size / (1024*1024):.2f} MB)")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_model_from_registry(model_name: str = None, stage: str = None) -> bool:
    """Download the model checkpoint from MLflow Model Registry using models:/.

    This resolves the latest version in the given stage, finds a .pth artifact
    in the associated run, and downloads it into model/transport_gnn_model.pth.
    """
    model_name = model_name or os.getenv("MODEL_NAME", "transport-gnn-model")
    stage = stage or os.getenv("MODEL_STAGE", "Production")

    # Prefer explicit MLflow env vars if set
    explicit_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()

    # DagsHub convenience envs
    dagshub_user = os.getenv('DAGSHUB_REPO_OWNER', 'iamsahan').strip()
    dagshub_repo = os.getenv('DAGSHUB_REPO_NAME', 'ml-services').strip()
    dagshub_token = os.getenv('DAGSHUB_TOKEN', '').strip()

    # Derive tracking URI
    tracking_uri = explicit_tracking_uri or f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"

    # Credentials precedence
    mlflow_user = os.getenv('MLFLOW_TRACKING_USERNAME', '').strip() or dagshub_user
    mlflow_pass = os.getenv('MLFLOW_TRACKING_PASSWORD', '').strip() or dagshub_token

    if mlflow_user and mlflow_pass:
        os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_user
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_pass

    mlflow.set_tracking_uri(tracking_uri)
    print(f"🔍 Connecting to MLflow Registry: {tracking_uri}")

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            print(f"⚠️  No versions found for model '{model_name}' in stage '{stage}'")
            return False

        ver = versions[0]
        run_id = ver.run_id
        print(f"📦 Resolved model: {model_name} {stage} -> run {run_id} (v{ver.version})")

        # Find a .pth artifact recursively
        artifact_path = _find_pth_artifact_recursive(client, run_id)
        if not artifact_path:
            print("⚠️  No .pth artifact found in the model run")
            return False

        model_dir = Path('model')
        model_dir.mkdir(exist_ok=True)
        print(f"⬇️  Downloading {artifact_path}...")
        local_path = client.download_artifacts(run_id, artifact_path, dst_path=str(model_dir))

        downloaded_file = Path(local_path)
        target_file = model_dir / 'transport_gnn_model.pth'
        if downloaded_file != target_file:
            downloaded_file.rename(target_file)
        
        # Check if it's a Git LFS pointer instead of actual file
        is_lfs, lfs_metadata = _is_git_lfs_pointer(target_file)
        if is_lfs:
            print("📌 Downloaded file is a Git LFS pointer, fetching actual binary...")
            
            # Get DagsHub credentials
            dagshub_user_lfs = os.getenv('DAGSHUB_REPO_OWNER', 'iamsahan').strip()
            dagshub_repo_lfs = os.getenv('DAGSHUB_REPO_NAME', 'ml-services').strip()
            dagshub_token_lfs = os.getenv('DAGSHUB_TOKEN', '').strip() or mlflow_pass
            
            if not _download_lfs_file(lfs_metadata, target_file, dagshub_user_lfs, dagshub_repo_lfs, dagshub_token_lfs):
                print("❌ Failed to download LFS file")
                return False

        # Validate like in other path
        if not target_file.exists():
            print("❌ Download failed: file not found")
            return False
        size = target_file.stat().st_size
        if size < 10 * 1024:
            print(f"❌ Downloaded file too small ({size} bytes) - likely not a model")
            try:
                target_file.unlink()
            except Exception:
                pass
            return False

        try:
            import torch
            torch.load(str(target_file), map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"❌ Downloaded file is not a valid PyTorch model: {e}")
            try:
                target_file.unlink()
            except Exception:
                pass
            return False

        print(f"✅ Model downloaded from registry to {target_file}")
        return True
    except Exception as e:
        print(f"❌ Error downloading from registry: {e}")
        import traceback
        traceback.print_exc()
        return False


def ensure_model_available_via_mlflow() -> bool:
    """Ensure the expected local checkpoint exists; try registry first, then experiment."""
    if check_local_model():
        return True
    # Try Model Registry path first
    if download_model_from_registry():
        return True
    # Fallback to experiment-based latest run
    return download_model_from_dagshub()


def check_local_model():
    """Check if a valid model already exists locally.

    We verify basic integrity to avoid picking up HTML/error files or empty files.
    """
    model_path = Path('model/transport_gnn_model.pth')

    if not model_path.exists():
        return False

    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Basic size sanity check (must be at least ~10KB)
    if size_bytes < 10 * 1024:
        print(f"❌ Existing model too small: {size_bytes} bytes — ignoring and re-downloading")
        try:
            model_path.unlink()
        except Exception:
            pass
        return False

    # Try to load as a PyTorch checkpoint (without executing arbitrary code)
    try:
        import torch
        torch.load(str(model_path), map_location='cpu', weights_only=False)
        print(f"✅ Model already exists: {model_path} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌ Existing model failed validation: {e} — will re-download")
        try:
            model_path.unlink()
        except Exception:
            pass
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ML Model Loader - Industry Standard MLOps Pattern")
    print("=" * 60)
    
    # Check if model already exists (e.g., baked into image or volume mount)
    if check_local_model():
        print("📌 Using existing model file")
        sys.exit(0)
    
    # Try to download from DagsHub MLflow
    print("\n🌐 Model not found locally, attempting download from DagsHub...")
    
    if download_model_from_dagshub():
        print("\n✅ Model ready for inference!")
        sys.exit(0)
    else:
        print("\n⚠️  Could not download model from DagsHub")
        print("💡 Options:")
        print("   1. Set DAGSHUB_TOKEN environment variable")
        print("   2. Include model in Docker image")
        print("   3. Mount model as volume: -v /path/to/model:/app/model")
        print("\n🔄 Container will continue - app will try to use local model if available")
        sys.exit(0)  # Exit successfully anyway, let app handle missing model
