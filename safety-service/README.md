# Safety Service (Travion ML Services)

FastAPI microservice for predicting tourist safety risk levels (Low/Medium/High) across multiple incident categories in Sri Lanka. Part of the Travion research project with DagsHub MLflow integration.

## 🎯 Features

- **Multi-output predictions** for seven incident categories (harassment, pickpocket, scam, bag_snatching, theft, extortion, money_theft)
- **DagsHub/MLflow integration** for model versioning and deployment
- **Model downloader** supporting MLflow Registry, GitHub Releases, or direct URLs
- **Modular routing** and batch prediction endpoints
- **Graceful fallback** heuristic when trained model is unavailable

## 🚀 Quick Start

### Local Development

```bash
# From safety-service folder
cd e:\Research\ml-services\safety-service

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your DagsHub credentials

# Start API
python app.py
# or
uvicorn app:app --reload --port 8003
```

Visit http://localhost:8003 for API info and http://localhost:8003/docs for Swagger UI.

## 📦 Model Source Options

### Option 1: MLflow Model Registry (DagsHub) - Recommended

```bash
# In .env file:
MODEL_SOURCE=mlflow
MLFLOW_TRACKING_URI=https://dagshub.com/your-org/ml-services.mlflow
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your-dagshub-token
MODEL_NAME=safety-risk-model
MODEL_STAGE=Production
```

**Benefits:**

- Version control for models
- Experiment tracking
- Model registry with staging/production
- Team collaboration via DagsHub

### Option 2: GitHub Releases (Fallback)

```bash
# In .env file:
MODEL_SOURCE=github_release
MODEL_REPO=Travion-Reserch-Project/ml-services
MODEL_RELEASE_TAG=latest
```

### Option 3: Direct URLs

```bash
# In .env file:
MODEL_SOURCE=url
SAFETY_MODEL_URL=https://your-storage.com/safety_risk_model.pkl
```

## 🧪 Training & Publishing Model

### 1. Train Model (Jupyter Notebook)

```python
# See notebooks/Tourist_Safety_Risk_Prediction.ipynb
# Run all cells including the export cell (cell 12)
```

### 2. Publish to DagsHub/MLflow

```bash
# Install MLflow
pip install mlflow

# Set DagsHub credentials
export MLFLOW_TRACKING_URI=https://dagshub.com/your-org/ml-services.mlflow
export MLFLOW_TRACKING_USERNAME=your-username
export MLFLOW_TRACKING_PASSWORD=your-token

# Log model to MLflow
python scripts/publish_model.py
```

### 3. Register Model in MLflow UI

- Go to your DagsHub MLflow UI
- Navigate to the run with your model
- Click "Register Model"
- Name: `safety-risk-model`
- Transition to "Production" stage

## 🐋 Docker

### Build

```bash
docker build -t travion-safety-service:latest .
```

### Run (with DagsHub)

```bash
docker run --rm -p 8003:8003 \
  -e MODEL_SOURCE=mlflow \
  -e MLFLOW_TRACKING_URI=https://dagshub.com/your-org/ml-services.mlflow \
  -e MLFLOW_TRACKING_USERNAME=your-username \
  -e MLFLOW_TRACKING_PASSWORD=your-token \
  -e MODEL_NAME=safety-risk-model \
  -e MODEL_STAGE=Production \
  travion-safety-service:latest
```

### Run (with GitHub Releases)

```bash
docker run --rm -p 8003:8003 \
  -e MODEL_SOURCE=github_release \
  -e MODEL_REPO=Travion-Reserch-Project/ml-services \
  -e MODEL_RELEASE_TAG=latest \
  travion-safety-service:latest
```

## 📡 API Endpoints

### Core Endpoints

- `GET /` – Service info
- `GET /api/safety/health` – Model/data health status
- `POST /api/safety/predict` – Single prediction
- `POST /api/safety/batch_predict` – Batch predictions
- `POST /api/safety/routes/batch_predict` – Alternative batch route (modular)

### Example Request

```json
POST /api/safety/predict
{
  "features": {
    "lat": 6.9271,
    "lon": 79.8612,
    "area_cluster": 0,
    "is_beach": 1,
    "is_crowded": 1,
    "is_tourist_place": 1,
    "is_transit": 0,
    "hour": 22,
    "day_of_week": 5,
    "is_weekend": 1,
    "police_nearby": 0
  }
}
```

### Example Response

```json
{
  "success": true,
  "prediction": {
    "risk_harassment": "High",
    "risk_pickpocket": "Medium",
    "risk_scam": "High",
    "risk_bag_snatching": "Medium",
    "risk_theft": "High",
    "risk_extortion": "Low",
    "risk_money_theft": "Medium"
  }
}
```

## 🧑‍💻 Development

### Run Tests

```bash
pytest -v tests/
```

### Project Structure

```
safety-service/
├── app.py                    # FastAPI application
├── utils/
│   ├── model_downloader.py  # MLflow/GitHub/URL downloader
│   └── safety_inference.py  # Prediction service
├── routes/
│   └── safety_routes.py     # Modular routes
├── tests/
│   └── test_app.py          # API tests
├── notebooks/
│   └── Tourist_Safety_Risk_Prediction.ipynb
├── model/                   # Downloaded models (gitignored)
├── data/                    # Training data (gitignored)
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 🔗 Integration with Travion

The safety-service integrates with:

- **travion-mobile**: SafetyAlert, MapScreen, ReportIncidentScreen, PoliceHelpScreen
- **travion-backend**: Aggregates safety data and incidents
- **infrastructure**: Kubernetes deployment configs

## 📝 Notes

### Model Artifact Format

Export your trained model as a single joblib dict:

```python
import joblib
artifact = {
    "model": trained_multioutput_classifier,
    "scaler": fitted_standard_scaler,
    "meta": {
        "feature_cols": FEATURE_COLS,
        "targets": TARGET_COLS,
        "labels": ["Low", "Medium", "High"]
    }
}
joblib.dump(artifact, "safety_risk_model.pkl")
```

Or separate files: `safety_risk_model.pkl`, `safety_scaler.pkl`, `safety_meta.json`

### DagsHub Setup

1. Create DagsHub account and join your team project
2. Get your token from DagsHub settings
3. Set environment variables in `.env`
4. Models will auto-download on first request

### Fallback Behavior

- If MLflow fails → fallback to GitHub Releases
- If no model available → use simple heuristic baseline
- Service always starts, even without models

## 🤝 Team

- **Safety Component Lead**: [Your Name]
- **Transport Service**: [Transport Lead]
- **Weather Service**: [Weather Lead]
- **Tour Plan Service**: [Tour Lead]

## 📄 License

Part of Travion Research Project
