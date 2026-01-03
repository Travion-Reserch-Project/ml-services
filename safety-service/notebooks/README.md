# Safety Service Notebooks

This directory contains Jupyter notebooks for model training and experimentation.

## Main Notebook

- `Tourist_Safety_Risk_Prediction.ipynb` - Complete training pipeline for safety risk model

## Instructions

1. Upload your notebook here after training
2. Run all cells including the export cell (last cell)
3. Download the exported artifacts:
   - safety_risk_model.pkl
   - safety_scaler.pkl
   - safety_meta.json
4. Publish to DagsHub/MLflow using `scripts/publish_model.py`
