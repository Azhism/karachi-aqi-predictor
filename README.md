# Karachi AQI Predictor

Air Quality Index prediction system for Karachi using Machine Learning and MLOps practices.

> 📄 **Complete Technical Documentation**: See [10Pearls_Internship.pdf](10Pearls_Internship.pdf) for comprehensive project report including challenges, learnings, and detailed analysis.

## Project Overview

This project predicts AQI category for Karachi, Pakistan for the next 72 hours using:
- Historical weather data (Open-Meteo API)
- Historical air quality data (OpenWeather API)
- Machine Learning models (Random Forest, XGBoost, LightGBM)
- Feature engineering pipeline
- Automated hourly data collection with GitHub Actions
- Automated daily model retraining
- Real-time dashboard with Streamlit Cloud

## Dataset

- **Location**: Karachi, Pakistan (24.8608°N, 67.0104°E)
- **Duration**: 79 days of historical data (Dec 1, 2025 - Feb 18, 2026)
- **Frequency**: Hourly measurements
- **Total Records**: 1,878 hourly data points (growing via automated collection)
- **Features**: 30 engineered features including:
  - **Raw Weather** (5): temperature, humidity, wind_speed, wind_direction, cloud_cover
  - **Raw Pollutants** (8): PM2.5, PM10, CO, NO, NO₂, O₃, SO₂, NH₃ (all in µg/m³)
  - **Target** (1): AQI category from OpenWeather API
  - **Time-based** (5): hour, day, month, day of week, cyclical encodings (sin/cos)
  - **Lag features** (3): AQI values at 24h, 48h, 72h ago (prevents data leakage)
  - **Rolling statistics** (6+): 72-hour window means and standard deviations
  - **Derived features** (8+): weather interactions, trend indicators

## Prediction Task

- **Type**: Multi-class Classification
- **Target**: AQI category (4 classes: 2, 3, 4, 5)
- **Prediction Horizon**: 72 hours ahead
- **Validation**: Temporal split (no shuffle to prevent data leakage)

## Architecture

```
┌─────────────────────────┐
│   Data Sources (APIs)   │
│  • Open-Meteo (Weather) │
│  • OpenWeather (AQI)    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Feature Pipeline      │
│   (GitHub Actions)      │
│   Runs: Every Hour      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   MongoDB Atlas         │
│   Collections:          │
│   • features (1,878)    │
│   • models (metadata)   │
│   • predictions         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Training Pipeline     │
│   (GitHub Actions)      │
│   Runs: Daily 3 AM UTC  │
│   Models: RF, XGB, LGBM │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Streamlit Cloud       │
│   • Model Registry      │
│   • 72h Predictions     │
│   • Historical Charts   │
│   • SHAP Analysis       │
└─────────────────────────┘
```

## 📁 Project Structure

```
karachi-aqi-predictor/
├── .github/workflows/          # CI/CD automation
│   ├── hourly-data-collection.yml
│   └── daily-model-training.yml
├── src/                        # Source code
│   ├── config.py              # Configuration management
│   ├── database.py            # MongoDB handler
│   ├── feature_pipeline.py    # Hourly data collection
│   ├── training_pipeline.py   # Model training
│   └── model_registry.py      # Model loading & predictions
├── models/                     # Saved models (.joblib)
│   ├── randomforest_model.joblib
│   ├── xgboost_model.joblib
│   ├── lightgbm_model.joblib
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   └── feature_columns.joblib
├── data/                       # Local CSV data
├── app.py                      # Streamlit dashboard (with SHAP)
├── show_latest.py              # Check latest MongoDB record
├── show_models.py              # View model registry from MongoDB
├── test_api_key.py             # Test API credentials
├── test_predictions.py         # Test model predictions
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (local)
└── README.md                   # This file
```

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/karachi-aqi-predictor.git
cd karachi-aqi-predictor
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:
```env
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
MONGODB_DATABASE=aqi_karachi
OPENWEATHER_API_KEY=your_api_key_here
CITY_NAME=Karachi
LATITUDE=24.8608
LONGITUDE=67.0104
```

### 4. Initial Setup (First Time Only)

The feature pipeline and training are automated via GitHub Actions. For local development:

```bash
# Collect initial data
python -m src.feature_pipeline

# Train models
python -m src.training_pipeline

# Run dashboard locally
streamlit run app.py
```

### 5. Deployment

The app is deployed on **Streamlit Cloud**:
- Connects to MongoDB Atlas
- Auto-updates from GitHub commits
- Uses secrets from Streamlit Cloud settings

## Model Performance

Current models trained on 1,800 samples (1,440 train / 360 test):

| Model | Test Accuracy | Train Accuracy | CV Accuracy (5-fold) | Notes |
|-------|---------------|----------------|---------------------|-------|
| **LightGBM** | **93.9%** | 100.0% | **93.1%** | Best performer 🏆 |
| **XGBoost** | 91.9% | 97.1% | 88.7% | Strong performance, slight overfitting |
| **RandomForest** | 65.6% | 69.1% | 64.2% | Conservative, lower variance |

**Model Registry**: 3 models in MongoDB with full metadata

**Validation Strategy**: Stratified random split for balanced class representation

**⚠️ Note on Accuracy**: These metrics use stratified random splitting. For realistic time-series validation (temporal split with no shuffle), expect lower performance as the model predicts truly unseen future data.

## Model Registry

The system maintains 3 trained models:

1. **LightGBM** - Best performer (93.9% test accuracy) 🏆
2. **XGBoost** - Strong second place (91.9% test accuracy)
3. **RandomForest** - Conservative baseline (65.6% test accuracy)

All models stored in:
- **Local**: `models/` directory (`.joblib` files)
- **MongoDB**: Complete metadata, hyperparameters, performance metrics, training timestamps

View all models and metrics: `python show_models.py`

## 🔍 Model Interpretability (SHAP)

The dashboard includes **SHAP (SHapley Additive exPlanations)** analysis to explain model predictions:

### Global Feature Importance
Shows which features matter most across all predictions:
- **Top Features** (by SHAP values):
  1. `cloud_cover` - Weather conditions
  2. `day_of_week_sin` - Temporal patterns
  3. `temperature` - Environmental factor
  4. `aqi_rolling_mean_72h` - Historical trend
  5. `aqi_lag_24h`, `aqi_lag_48h` - Recent history

### Individual Prediction Explanation
For each prediction, SHAP shows:
- Which features **increase** AQI (positive SHAP values)
- Which features **decrease** AQI (negative SHAP values)
- Magnitude of each feature's impact

**Benefits:**
- 🧠 Understand why the model predicts specific AQI values
- 🔎 Identify key driving factors (pollution, weather, time)
- ✅ Build trust in model decisions
- 📊 Validate that the model uses sensible patterns

**Technology:** Uses TreeExplainer for tree-based models (RandomForest, XGBoost, LightGBM)

## Automation & MLOps

### GitHub Actions Workflows

1. **Hourly Data Collection** (`hourly-data-collection.yml`)
   - Runs: Every hour at :00
   - Fetches weather + AQI data from APIs
   - Engineers features
   - Stores in MongoDB
   
2. **Daily Model Training** (`daily-model-training.yml`)
   - Runs: Daily at 2:00 AM UTC
   - Trains all 3 models
   - Evaluates performance
   - Updates model registry
   - Commits updated models to Git

### Features

- ✅ Automated data collection (hourly)
- ✅ Automated model retraining (daily)
- ✅ Model versioning in MongoDB
- ✅ Temporal validation (no data leakage)
- ✅ Production deployment on Streamlit Cloud
- ✅ **SHAP model interpretability** (feature importance + prediction explanations)

## �️ Utilities

**Testing & Debugging:**
- `test_api_key.py` - Verify OpenWeather API credentials
- `test_predictions.py` - Test model predictions locally
- `show_latest.py` - Check latest record in MongoDB
- `show_models.py` - View all trained models and performance from MongoDB

**Manual Operations:**
```bash
# Check latest data
python show_latest.py

# View model registry
python show_models.py

# Test API connection
python test_api_key.py

# Test model predictions
python test_predictions.py
```

## Author

**Azhab Safwan Babar**
