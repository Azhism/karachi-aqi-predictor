"""
Configuration file for Karachi AQI Predictor
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'aqi_karachi')

# Location Configuration
CITY_NAME = os.getenv('CITY_NAME', 'Karachi')
LATITUDE = float(os.getenv('LATITUDE', 24.8608))
LONGITUDE = float(os.getenv('LONGITUDE', 67.0104))

# Model Configuration
PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', 72))  # Hours to predict
TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', 0.2))
RANDOM_STATE = 42

# Feature Engineering Configuration
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 72]
ROLLING_WINDOWS = [6, 12, 24, 48, 72]

# MongoDB Collections
COLLECTION_RAW_DATA = 'raw_data'
COLLECTION_FEATURES = 'features'
COLLECTION_PREDICTIONS = 'predictions'
COLLECTION_MODELS = 'models'

# Target variable
TARGET_VARIABLE = 'aqi'

# Features to use for training (we'll populate this after data analysis)
FEATURE_COLUMNS = None  # Will be set dynamically

print("✅ Configuration loaded successfully")
