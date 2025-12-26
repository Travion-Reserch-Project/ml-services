"""
Script to publish trained model to DagsHub/MLflow Model Registry
Run after training and exporting model artifacts

Usage:
    python scripts/publish_model.py --artifact safety_risk_model.pkl

Requirements:
    - Trained model exported from notebook
    - DagsHub credentials set in .env or environment
"""
import os
import sys
import argparse
import joblib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    print("❌ MLflow not installed. Run: pip install mlflow")
    sys.exit(1)


def publish_model(
    artifact_path: str,
    model_name: str = "safety-risk-model",
    run_name: str = "safety-production",
    description: str = "Tourist safety risk prediction model"
):
    """
    Publish model to MLflow Model Registry
    
    Args:
        artifact_path: Path to unified model artifact (.pkl file)
        model_name: Name to register in MLflow
        run_name: Name for this MLflow run
        description: Model description
    """
    # Set MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("❌ MLFLOW_TRACKING_URI not set in environment")
        print("   Set it in .env: MLFLOW_TRACKING_URI=https://dagshub.com/your-org/ml-services.mlflow")
        sys.exit(1)
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"📊 MLflow Tracking URI: {tracking_uri}")
    
    # Set credentials
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        print(f"✓ Credentials set for user: {username}")
    
    # Load artifact
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        print(f"❌ Artifact not found: {artifact_path}")
        sys.exit(1)
    
    print(f"📦 Loading artifact: {artifact_path}")
    artifact = joblib.load(artifact_path)
    
    if isinstance(artifact, dict):
        model = artifact.get("model")
        scaler = artifact.get("scaler")
        meta = artifact.get("meta", {})
        target_encoders = artifact.get("target_encoders")
    else:
        model = artifact
        scaler = None
        meta = {}
        target_encoders = None
    
    if model is None:
        print("❌ No model found in artifact")
        sys.exit(1)
    
    print("✓ Model loaded successfully")
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        print(f"\n🚀 Starting MLflow run: {run_name}")
        
        # Log parameters
        if meta:
            mlflow.log_param("feature_cols", meta.get("feature_cols", []))
            mlflow.log_param("targets", meta.get("targets", []))
            mlflow.log_param("labels", meta.get("labels", []))
        
        # Log model
        print("📤 Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Log scaler if available
        if scaler is not None:
            print("📤 Logging scaler...")
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler"
            )
        
        # Log metadata
        if meta:
            print("📤 Logging metadata...")
            import json
            import tempfile
            # Use system temp directory (cross-platform)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                meta_file = f.name
                json.dump(meta, f, indent=2)
            mlflow.log_artifact(meta_file, "metadata")
            # Clean up temp file
            os.unlink(meta_file)
        
        if target_encoders is not None:
            import tempfile
            import pickle
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                enc_file = f.name
                pickle.dump(target_encoders, f)
            mlflow.log_artifact(enc_file, "target_encoders")
            os.unlink(enc_file)
            print("📤 Logged target_encoders artifact")

        # Add description
        mlflow.set_tag("mlflow.note.content", description)
        mlflow.set_tag("service", "safety-service")
        mlflow.set_tag("framework", "scikit-learn")
        
        run_id = mlflow.active_run().info.run_id
        print(f"✅ Model published successfully!")
        print(f"   Run ID: {run_id}")
        print(f"   Model Name: {model_name}")
        print(f"\n🔗 View in DagsHub:")
        print(f"   {tracking_uri.replace('.mlflow', '')}")
        print(f"\n📝 Next steps:")
        print(f"   1. Go to MLflow UI and navigate to Models → {model_name}")
        print(f"   2. Select the latest version")
        print(f"   3. Click 'Transition to' → 'Production'")
        print(f"   4. Update safety-service .env with:")
        print(f"      MODEL_SOURCE=mlflow")
        print(f"      MODEL_NAME={model_name}")
        print(f"      MODEL_STAGE=Production")


def main():
    parser = argparse.ArgumentParser(description="Publish model to DagsHub/MLflow")
    parser.add_argument(
        "--artifact",
        default="safety_risk_model.pkl",
        help="Path to model artifact (default: safety_risk_model.pkl)"
    )
    parser.add_argument(
        "--name",
        default="safety-risk-model",
        help="Model name in registry (default: safety-risk-model)"
    )
    parser.add_argument(
        "--run-name",
        default="safety-production",
        help="MLflow run name (default: safety-production)"
    )
    parser.add_argument(
        "--description",
        default="Tourist safety risk prediction model",
        help="Model description"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PUBLISHING MODEL TO DAGSHUB/MLFLOW")
    print("=" * 60)
    
    publish_model(
        artifact_path=args.artifact,
        model_name=args.name,
        run_name=args.run_name,
        description=args.description
    )
    
    print("\n" + "=" * 60)
    print("✅ PUBLISHING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
