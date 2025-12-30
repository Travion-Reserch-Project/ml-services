"""
Script to publish trained UV Risk model to DagsHub / MLflow Model Registry

Usage:
    python scripts/publish_model.py \
        --model models/uv_risk_gradient_boosting.pkl \
        --features models/model_features.pkl \
        --decode-map models/risk_decode_map.pkl

Requirements:
    - Model exported from notebook
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
    model_path: str,
    features_path: str,
    decode_map_path: str,
    model_name: str = "uv-risk-model",
    run_name: str = "uv-risk-production",
    description: str = "UV exposure risk prediction model for tourists"
):
    """
    Publish model and metadata to MLflow Model Registry
    """

    # ------------------------------
    # Set MLflow tracking URI
    # ------------------------------
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("❌ MLFLOW_TRACKING_URI not set")
        print("   Example:")
        print("   MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    print(f"📊 MLflow Tracking URI: {tracking_uri}")

    # ------------------------------
    # Set credentials
    # ------------------------------
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        print(f"✓ Credentials set for user: {username}")

    # ------------------------------
    # Load artifacts
    # ------------------------------
    model_path = Path(model_path)
    features_path = Path(features_path)
    decode_map_path = Path(decode_map_path)

    for path in [model_path, features_path, decode_map_path]:
        if not path.exists():
            print(f"❌ File not found: {path}")
            sys.exit(1)

    print("📦 Loading artifacts...")
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    decode_map = joblib.load(decode_map_path)

    print("✓ Model and metadata loaded")

    # ------------------------------
    # Start MLflow run
    # ------------------------------
    with mlflow.start_run(run_name=run_name):
        print(f"\n🚀 MLflow Run Started: {run_name}")

        # Log parameters
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("risk_classes", list(decode_map.values()))

        # Log model
        print("📤 Logging model...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

        # Log feature list
        print("📤 Logging feature metadata...")
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"features": feature_cols}, f, indent=2)
            feature_file = f.name

        mlflow.log_artifact(feature_file, "metadata")
        os.unlink(feature_file)

        # Log risk decode map
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(decode_map, f, indent=2)
            decode_file = f.name

        mlflow.log_artifact(decode_file, "metadata")
        os.unlink(decode_file)

        # Tags
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.set_tag("task", "classification")
        mlflow.set_tag("domain", "tourist-safety")
        mlflow.set_tag("mlflow.note.content", description)

        run_id = mlflow.active_run().info.run_id

        print("\n✅ MODEL PUBLISHED SUCCESSFULLY")
        print(f"Run ID      : {run_id}")
        print(f"Model Name  : {model_name}")
        print("\n🔗 View on DagsHub:")
        print(tracking_uri.replace(".mlflow", ""))

        print("\n📝 Next Steps:")
        print(f"1. Open MLflow → Models → {model_name}")
        print("2. Select latest version")
        print("3. Transition stage → Production")


def main():
    parser = argparse.ArgumentParser(description="Publish UV Risk Model to DagsHub/MLflow")

    parser.add_argument("--model", default="models/uv_risk_model.pkl")
    parser.add_argument("--features", default="models/model_features.pkl")
    parser.add_argument("--decode-map", default="models/risk_decode_map.pkl")
    parser.add_argument("--name", default="uv-risk-model")
    parser.add_argument("--run-name", default="uv-risk-production")
    parser.add_argument("--description", default="UV exposure risk prediction model for tourists")

    args = parser.parse_args()

    print("=" * 60)
    print("PUBLISHING UV RISK MODEL TO DAGSHUB")
    print("=" * 60)

    publish_model(
        model_path=args.model,
        features_path=args.features,
        decode_map_path=args.decode_map,
        model_name=args.name,
        run_name=args.run_name,
        description=args.description
    )

    print("\n" + "=" * 60)
    print("✅ PUBLISHING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
