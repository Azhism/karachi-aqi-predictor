# Karachi AQI Predictor

Air Quality Index prediction system for Karachi using Machine Learning and MLOps practices.

## 🎯 Project Overview

This project predicts PM2.5 levels (Air Quality Index) for Karachi, Pakistan for the next 72 hours using:
- Historical weather data (Open-Meteo API)
- Historical air quality data (Open-Meteo Air Quality API)
- Machine Learning models (Random Forest, XGBoost, LightGBM)
- Feature engineering pipeline
- Automated retraining with GitHub Actions
- Real-time dashboard with Streamlit

## 📊 Dataset

- **Location**: Karachi, Pakistan (24.8608°N, 67.0104°E)
- **Duration**: 180 days of historical data
- **Frequency**: Hourly measurements
- **Features**: 66+ engineered features including:
  - Weather: temperature, humidity, wind speed, cloud cover, precipitation
  - Pollution: PM2.5, PM10, CO, NO2, SO2, O3
  - Time-based: hour, day, month, cyclical encodings
  - Lag features: 1h, 3h, 6h, 12h, 24h, 48h, 72h
  - Rolling features: means, stds, mins, maxs
  - Derived features: ratios, interactions, changes

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Sources   │
│  (Open-Meteo)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Pipeline│
│  (Hourly Run)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    MongoDB      │
│ Feature Store   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Training Pipeline│
│  (Daily Run)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit App   │
│  (Predictions)  │
└─────────────────┘
```

## 📁 Project Structure

```
karachi-aqi-predictor/
├── data/                          # Local data storage
├── notebooks/                     # Jupyter notebooks
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration
│   ├── database.py               # MongoDB handler
│   ├── feature_pipeline.py       # Feature engineering
│   └── training_pipeline.py      # Model training
├── models/                        # Saved models
├── .github/workflows/             # CI/CD automation
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Dependencies
└── .env                          # Environment variables
```

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/karachi-aqi-predictor.git
cd karachi-aqi-predictor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Edit `.env` file with your credentials:
```env
# Replace with your actual MongoDB Atlas connection string
MONGODB_URI=mongodb+srv://<username>:<password>@<your-cluster>.mongodb.net/<database>
```

### 4. Upload Initial Data
```bash
python upload_to_mongodb.py
```

### 5. Run Feature Pipeline
```bash
python src/feature_pipeline.py
```

### 6. Train Model
```bash
python src/training_pipeline.py
```

### 7. Run Dashboard
```bash
streamlit run app.py
```

## 📊 Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD |

## 🔄 Automation

- **Feature Pipeline**: Runs hourly via GitHub Actions
- **Training Pipeline**: Runs daily via GitHub Actions
- **Model Registry**: MongoDB stores model metadata and metrics

## 👨‍💻 Author

Muhammad Mobeen (Instructor: 10 Pearls)

## 📝 License

This project is for educational purposes.
