import pandas as pd

df = pd.read_csv('data/karachi_aqi_direct_dataset.csv', nrows=1)

print("\n📊 YOUR DATASET COLUMNS:")
print("="*60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n✅ Total columns: {len(df.columns)}")

# Check for engineered features
has_lag = any('lag' in col for col in df.columns)
has_rolling = any('rolling' in col for col in df.columns)
has_time = any(col in ['hour_sin', 'hour_cos', 'month_sin'] for col in df.columns)

print("\n🔍 Feature Engineering Status:")
print(f"   Lag features: {'✅ YES' if has_lag else '❌ NO'}")
print(f"   Rolling features: {'✅ YES' if has_rolling else '❌ NO'}")
print(f"   Time features: {'✅ YES' if has_time else '❌ NO'}")

if has_lag and has_rolling and has_time:
    print("\n🎉 Your data is ALREADY FULLY FEATURED!")
    print("   You can skip the feature pipeline and go straight to training!")
else:
    print("\n⚠️  Your data needs feature engineering.")
    print("   Run the feature pipeline to add missing features.")
