#!/usr/bin/env python3
"""
Download trained model from DagsHub MLflow artifact store
Industry standard approach: Separate model training from model serving

Enhancements:
- Respect MLFLOW_TRACKING_URI/USERNAME/PASSWORD if provided
- Fallback to DAGSHUB_REPO_OWNER/NAME + DAGSHUB_TOKEN
- Recursively search run artifacts to find a .pth file
"""
import os
import sys
from pathlib import Path
import mlflow


def _find_pth_artifact_recursive(client: mlflow.tracking.MlflowClient, run_id: str, path: str = "") -> str:
    """Recursively search artifacts for a .pth file and return its artifact path."""
    items = client.list_artifacts(run_id, path) if path else client.list_artifacts(run_id)
    for it in items:
        # If directory, recurse
        if it.is_dir:
            found = _find_pth_artifact_recursive(client, run_id, it.path)
            if found:
                return found
        else:
            if it.path.endswith('.pth'):
                return it.path
    return ""


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
        # Get the experiment
        experiment = mlflow.get_experiment_by_name('transport-gnn')
        
        if not experiment:
            print("⚠️  Experiment 'transport-gnn' not found")
            return False
        
        # Get the latest run with the model artifact
        client = mlflow.tracking.MlflowClient()
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
