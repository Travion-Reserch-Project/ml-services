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
        
        print(f"✅ Model downloaded successfully to {target_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_local_model():
    """Check if model already exists locally."""
    model_path = Path('model/transport_gnn_model.pth')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model already exists: {model_path} ({size_mb:.2f} MB)")
        return True
    
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
        sys.exit(1)
