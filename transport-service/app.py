from fastapi import FastAPI, Query
import pickle
import numpy as np

app = FastAPI(title="Safety Risk Prediction Service")

# load dummy model
model = pickle.load(open("model/risk_model.pkl", "rb"))

@app.get("/")
def root():
    return {"message": "Safety Risk ML Service is running"}

@app.get("/predict")
def predict(lat: float = Query(...), lon: float = Query(...)):
    # dummy prediction logic
    risk_score = np.random.rand()
    level = "High" if risk_score > 0.6 else "Medium" if risk_score > 0.3 else "Low"
    return {"latitude": lat, "longitude": lon, "risk_score": risk_score, "risk_level": level}
