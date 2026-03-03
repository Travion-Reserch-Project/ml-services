#!/usr/bin/env python3
"""Test DagsHub MLflow connection and list experiments/runs"""
import os
from dotenv import load_dotenv
import mlflow

# Load environment
load_dotenv()

# Get tracking URI
tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')

print(f"🔗 Tracking URI: {tracking_uri}")
print(f"👤 Username: {username}")
print(f"🔑 Password: {'*' * len(password) if password else 'NOT SET'}")

# Set tracking URI
mlflow.set_tracking_uri(tracking_uri)
os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = password

try:
    client = mlflow.tracking.MlflowClient()
    
    # List experiments
    experiments = client.search_experiments()
    print(f"\n📊 Found {len(experiments)} experiments:")
    
    for exp in experiments:
        print(f"\n  Experiment: {exp.name} (ID: {exp.experiment_id})")
        
        # List runs in this experiment
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=5
        )
        
        print(f"  📈 Runs: {len(runs)}")
        for run in runs:
            print(f"    - Run ID: {run.info.run_id[:8]}... | Status: {run.info.status}")
            print(f"      Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
            
            # List artifacts
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"      Artifacts: {[a.path for a in artifacts]}")
    
    print("\n✅ Connection successful!")
    
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()
