# Pillar 2: CrowdCast Engine

## Research Problem
**Overtourism**: Tourists visit peak locations at peak times, ruining the experience.

## Solution Logic
**Spatiotemporal Prediction**: Uses Random Forest on Google Trends & Calendar data to forecast crowd density (0-100%).

## Technology
- **Model**: Random Forest Regressor / XGBoost
- **Performance**: R² = 0.9982, RMSE = 0.0045

## Features
| Feature | Description |
|---------|-------------|
| `month` | 1-12 (Seasonality) |
| `day_of_week` | 0-6 (Weekend effect) |
| `hour` | 0-23 (Time of day) |
| `is_poya_holiday` | Poya day (0/1) |
| `is_school_holiday` | School break (0/1) |
| `google_trend_30d` | Search interest (0-100) |
| `loc_type` | Heritage/Beach/Nature/Religious/Urban |

## Target
`crowd_level` (0.0 - 1.0)

## Status: IMPLEMENTED (Model trained, API pending)

## Files
```
pillar_2_crowdcast/
├── notebooks/
│   └── crowdcast_model_training.ipynb  # Training notebook
├── data/
│   ├── crowd_data_pipeline.py          # Data scraping pipeline
│   ├── crowdcast_dataset.csv           # 3.5M training samples
│   ├── google_trends_raw.csv           # Google Trends data
│   ├── popular_times_default.json      # Hourly patterns
│   └── holidays/
│       ├── 2021-2025.csv               # Year-wise holidays
│       └── consolidated_holidays.csv   # All holidays
└── models/
    ├── crowdcast_model.joblib          # Random Forest model
    ├── crowdcast_model.json            # JSON export
    ├── label_encoder.joblib            # Location type encoder
    └── metadata.json                   # Performance metrics
```
