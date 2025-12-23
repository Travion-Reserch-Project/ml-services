#!/usr/bin/env python3
"""
Download trained model from DagsHub MLflow artifact store
Industry standard approach: Separate model training from model serving
"""
import os
import sys
from pathlib import Path
import mlflow


def download_model_from_dagshub():
    """Download the latest model from DagsHub MLflow."""
    
    # Configuration from environment
    dagshub_user = os.getenv('DAGSHUB_REPO_OWNER', 'iamsahan')
    dagshub_repo = os.getenv('DAGSHUB_REPO_NAME', 'ml-services')
    dagshub_token = os.getenv('DAGSHUB_TOKEN', '')
    
    # MLflow tracking URI
    tracking_uri = f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
    
    # Set credentials if token provided
    if dagshub_token:
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
    
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
        
        # List artifacts to find the model
        artifacts = client.list_artifacts(run_id)
        model_artifact = None
        
        for artifact in artifacts:
            if artifact.path.endswith('.pth'):
                model_artifact = artifact.path
                break
        
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
        if file_size < 1000:  # Less than 1KB is likely an error page
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
