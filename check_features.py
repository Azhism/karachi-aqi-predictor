"""Check which features were used in training"""
import joblib

# Load feature columns from the trained model
feature_cols = joblib.load('models/feature_columns.joblib')

print(f"📊 Training used {len(feature_cols)} features:\n")
for i, col in enumerate(feature_cols, 1):
    print(f"{i:2d}. {col}")

# Check for problematic columns
problematic = ['precipitation', 'aqi_change_1h', 'aqi_change_3h']
print(f"\n🔍 Checking for problematic columns:")
for col in problematic:
    if col in feature_cols:
        print(f"   ⚠️  {col} IS USED in training")
    else:
        print(f"   ✅ {col} NOT USED in training")
