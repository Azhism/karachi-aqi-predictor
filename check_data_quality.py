import pandas as pd
import numpy as np

print("="*60)
print("🔍 DIAGNOSING DATA QUALITY")
print("="*60)

# Load CSV
df = pd.read_csv('data/karachi_aqi_direct_dataset.csv')

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"Records: {len(df):,}")
print(f"Features: {len(df.columns)}")

# Check for target column
print(f"\n🎯 Target Column Check:")
print(f"'aqi' exists: {'aqi' in df.columns}")
if 'aqi' in df.columns:
    print(f"AQI range: {df['aqi'].min():.2f} to {df['aqi'].max():.2f}")
    print(f"AQI mean: {df['aqi'].mean():.2f}")
    print(f"AQI std: {df['aqi'].std():.2f}")
    print(f"AQI nulls: {df['aqi'].isnull().sum()}")

# Check for missing values
print(f"\n❓ Missing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✅ No missing values!")

# Check for infinite values
print(f"\n∞ Infinite Values:")
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"Total infinite values: {inf_count}")

# Check variance
print(f"\n📈 Feature Variance Check:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
zero_var = [col for col in numeric_cols if df[col].std() == 0]
if zero_var:
    print(f"⚠️  Columns with ZERO variance: {zero_var}")
else:
    print("✅ All features have variance!")

# Sample predictions
print(f"\n🔮 Target Distribution:")
print(df['aqi'].describe())

# Check for data leakage (future data in features)
print(f"\n⚠️  Potential Data Leakage Check:")
feature_cols = [col for col in df.columns if 'lag' in col.lower() or 'rolling' in col.lower()]
print(f"Lag/Rolling features found: {len(feature_cols)}")
print(f"Sample: {feature_cols[:5]}")

# Check unique values
print(f"\n🔢 Target Variable Analysis:")
unique_aqi = df['aqi'].unique()
print(f"Unique AQI values: {len(unique_aqi)}")
print(f"AQI value counts:\n{df['aqi'].value_counts().sort_index()}")

print("\n" + "="*60)
