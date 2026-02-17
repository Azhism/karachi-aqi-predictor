"""Check for NaN values in MongoDB features collection"""
from src.database import MongoDBHandler
import pandas as pd
import numpy as np

# Connect to database
db = MongoDBHandler()

# Load features
print("📥 Loading features from MongoDB...")
records = list(db.features.find().sort("timestamp", 1))
print(f"✅ Loaded {len(records):,} records\n")

if records:
    df = pd.DataFrame(records)
    df = df.drop(['_id'], axis=1, errors='ignore')
    
    print("📊 Data shape:", df.shape)
    print(f"\n📋 Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # Try to find timestamp column
    timestamp_col = None
    for col in df.columns:
        if 'time' in col.lower():
            timestamp_col = col
            break
    
    if timestamp_col:
        print(f"Date range: {df[timestamp_col].min()} to {df[timestamp_col].max()}\n")
    else:
        print("⚠️ No timestamp column found\n")
    
    # Check for NaN values in each column
    print("🔍 NaN counts by column:")
    nan_counts = df.isna().sum()
    nan_pct = (nan_counts / len(df) * 100).round(2)
    
    # Show columns with NaN
    has_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(has_nan) > 0:
        print(f"\n⚠️  Found {len(has_nan)} columns with NaN values:")
        for col, count in has_nan.items():
            pct = nan_pct[col]
            print(f"   {col:30s}: {count:5,} records ({pct:6.2f}%)")
    else:
        print("   ✅ No NaN values found!")
    
    # Check total records with ANY NaN
    records_with_nan = df.isna().any(axis=1).sum()
    print(f"\n📉 Records with ANY NaN: {records_with_nan:,} / {len(df):,} ({records_with_nan/len(df)*100:.2f}%)")
    
    # Check complete records
    complete_records = len(df.dropna())
    print(f"✅ Complete records (no NaN): {complete_records:,} / {len(df):,} ({complete_records/len(df)*100:.2f}%)")
    
    # Sample first few records
    print("\n📋 Sample of first 5 records:")
    sample_cols = [timestamp_col, 'aqi', 'temperature_2m'] if timestamp_col else ['aqi', 'temperature_2m']
    # Add lag columns if they exist
    for col in ['aqi_lag_1h', 'aqi_lag_72h']:
        if col in df.columns:
            sample_cols.append(col)
    print(df.head()[sample_cols].to_string())

db.close()
