"""
Debug the feature pipeline to see where records are being filtered out
"""
from src.feature_pipeline import FeaturePipeline
import pandas as pd

fp = FeaturePipeline()

print("="*70)
print("🔍 PIPELINE DEBUG")
print("="*70)

# Fetch data
weather_df = fp.fetch_weather_data(days=2)
aqi_df = fp.fetch_aqi_data(days=2)

print(f"\n📊 Fetched data:")
print(f"   Weather records: {len(weather_df)}")
print(f"   AQI records: {len(aqi_df)}")
print(f"   Weather date range: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
print(f"   AQI date range: {aqi_df['datetime'].min()} to {aqi_df['datetime'].max()}")

# Combine
df_new = pd.merge(weather_df, aqi_df, on='datetime', how='inner')
df_new['datetime'] = pd.to_datetime(df_new['datetime'])
print(f"\n🔗 After merge: {len(df_new)} records")
print(f"   Date range: {df_new['datetime'].min()} to {df_new['datetime'].max()}")

# Get existing features
df_existing = fp.db.get_features(limit=200)
print(f"\n📥 Existing features from DB: {len(df_existing)} records")
if not df_existing.empty:
    print(f"   Date range: {df_existing['datetime'].min()} to {df_existing['datetime'].max()}")

# Combine
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['datetime'])
df_combined = df_combined.sort_values('datetime').reset_index(drop=True)
print(f"\n🔗 After combining: {len(df_combined)} records")
print(f"   Date range: {df_combined['datetime'].min()} to {df_combined['datetime'].max()}")

# Engineer features
df_combined = fp.create_time_features(df_combined)
df_combined = fp.create_lag_features(df_combined)
df_combined = fp.create_rolling_features(df_combined)
df_combined = fp.create_derived_features(df_combined)

print(f"\n🔧 After feature engineering: {len(df_combined)} records")
print(f"   Columns: {len(df_combined.columns)}")

# Check NaN
print(f"\n🧹 NaN analysis:")
print(f"   Records before dropna: {len(df_combined)}")
print(f"   Records with ANY NaN: {df_combined.isna().any(axis=1).sum()}")

df_clean = df_combined.dropna()
print(f"   Records after dropna: {len(df_clean)}")

if len(df_clean) > 0:
    print(f"   Date range after dropna: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")

# Check what's new
if not df_existing.empty:
    latest_db_time = pd.to_datetime(df_existing['datetime'].max())
    print(f"\n📌 Latest DB time: {latest_db_time}")
    df_new_only = df_clean[df_clean['datetime'] > latest_db_time]
    print(f"   Records newer than DB: {len(df_new_only)}")
    
    if len(df_new_only) > 0:
        print(f"   New records date range: {df_new_only['datetime'].min()} to {df_new_only['datetime'].max()}")
    else:
        print(f"   Latest clean record: {df_clean['datetime'].max()}")
        print(f"   Difference: Latest clean ({df_clean['datetime'].max()}) vs DB ({latest_db_time})")
