"""
Reload CSV data with clean features (no data leakage)
"""
import pandas as pd
import numpy as np
from src.database import MongoDBHandler
from src.config import LAG_FEATURES, ROLLING_WINDOWS

print("="*60)
print("🧹 CLEANING AND RELOADING DATA (NO LEAKAGE)")
print("="*60 )

# Load CSV
print("\n📂 Loading CSV...")
df = pd.read_csv('data/karachi_aqi_direct_dataset.csv')
print(f"✅ Loaded {len(df):,} records")

# Keep only RAW data columns (no engineered features)
raw_columns = [
    'datetime', 'temperature', 'humidity', 'wind_speed', 'wind_direction',
    'cloud_cover', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
    'aqi', 'aqi_category'
]

print(f"\n🔪 Keeping only {len(raw_columns)} raw columns...")
df_raw = df[raw_columns].copy()
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
df_raw = df_raw.sort_values('datetime').reset_index(drop=True)

# Regenerate clean features
print("\n🏗️  Regenerating features WITHOUT leakage...")

# Time features
df_raw['year'] = df_raw['datetime'].dt.year
df_raw['month'] = df_raw['datetime'].dt.month
df_raw['day'] = df_raw['datetime'].dt.day
df_raw['hour'] = df_raw['datetime'].dt.hour
df_raw['day_of_week'] = df_raw['datetime'].dt.dayofweek
df_raw['is_weekend'] = (df_raw['day_of_week'] >= 5).astype(int)
df_raw['week_of_year'] = df_raw['datetime'].dt.isocalendar().week

# Cyclical features
df_raw['hour_sin'] = np.sin(2 * np.pi * df_raw['hour'] / 24)
df_raw['hour_cos'] = np.cos(2 * np.pi * df_raw['hour'] / 24)
df_raw['day_of_week_sin'] = np.sin(2 * np.pi * df_raw['day_of_week'] / 7)
df_raw['day_of_week_cos'] = np.cos(2 * np.pi * df_raw['day_of_week'] / 7)
df_raw['month_sin'] = np.sin(2 * np.pi * df_raw['month'] / 12)
df_raw['month_cos'] = np.cos(2 * np.pi * df_raw['month'] / 12)

print(f"   ✅ Created 13 time features")

# Lag features (ONLY valid ones: 24h, 48h, 72h)
print(f"\n   Creating lag features for: {LAG_FEATURES}")
for lag in LAG_FEATURES:
    df_raw[f'aqi_lag_{lag}h'] = df_raw['aqi'].shift(lag)

df_raw['temperature_lag_24h'] = df_raw['temperature'].shift(24)
df_raw['humidity_lag_24h'] = df_raw['humidity'].shift(24)
df_raw['wind_speed_lag_24h'] = df_raw['wind_speed'].shift(24)

print(f"   ✅ Created {len(LAG_FEATURES) + 3} lag features")

# Rolling features (ONLY valid ones: 24h, 48h, 72h)
print(f"\n   Creating rolling features for: {ROLLING_WINDOWS}")
for window in ROLLING_WINDOWS:
    df_raw[f'aqi_rolling_mean_{window}h'] = df_raw['aqi'].rolling(window).mean()
    df_raw[f'aqi_rolling_std_{window}h'] = df_raw['aqi'].rolling(window).std()
    df_raw[f'aqi_rolling_min_{window}h'] = df_raw['aqi'].rolling(window).min()
    df_raw[f'aqi_rolling_max_{window}h'] = df_raw['aqi'].rolling(window).max()

print(f"   ✅ Created {len(ROLLING_WINDOWS) * 4} rolling features")

# Derived features
df_raw['aqi_change_24h'] = df_raw['aqi'].diff(24)
df_raw['temp_humidity'] = df_raw['temperature'] * df_raw['humidity']
df_raw['wind_pollution'] = df_raw['wind_speed'] * df_raw['aqi']
df_raw['temp_wind'] = df_raw['temperature'] * df_raw['wind_speed']

print(f"   ✅ Created 4 derived features")

print(f"\n📊 Final dataset:")
print(f"   Records: {len(df_raw):,}")
print(f"   Features: {len(df_raw.columns)}")
print(f"   Date range: {df_raw['datetime'].min()} to {df_raw['datetime'].max()}")

# Upload to MongoDB
print("\n🔌 Connecting to MongoDB...")
db = MongoDBHandler()

print("\n💾 Uploading clean features to MongoDB...")
db.insert_features(df_raw)

print("\n✅ Verifying...")
db.get_collection_stats()

db.close()

print("\n" + "="*60)
print("✅ CLEAN DATA RELOAD COMPLETE!")
print("="*60)
print(f"\n🚫 Removed leaky features:")
print(f"   • aqi_lag_1h, 2h, 3h, 6h, 12h")
print(f"   • aqi_rolling_*_6h, 12h")
print(f"\n✅ Kept valid features:")
print(f"   • aqi_lag_24h, 48h, 72h")
print(f"   • aqi_rolling_*_24h, 48h, 72h")
print(f"\n💡 Now you can train models on clean data:")
print(f"   python -m src.training_pipeline")
