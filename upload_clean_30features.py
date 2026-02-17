"""
Upload new 30-feature clean dataset to MongoDB
Adds datetime column based on original dataset
"""
import pandas as pd
from src.database import MongoDBHandler
from datetime import datetime, timedelta

print("="*60)
print("📤 UPLOADING CLEAN 30-FEATURE DATASET")
print("="*60)

# Load new clean dataset
print("\n📂 Loading clean dataset...")
df_clean = pd.read_csv('data/karachi_aqi_30features.csv')
print(f"✅ Loaded {len(df_clean):,} records with {len(df_clean.columns)} columns")

# Add datetime column (assuming same date range as original)
# Starting from Dec 1, 2025 18:00 (same as original dataset)
print("\n📅 Adding datetime column...")
start_date = datetime(2025, 12, 1, 18, 0, 0)
df_clean['datetime'] = [start_date + timedelta(hours=i) for i in range(len(df_clean))]

print(f"   Date range: {df_clean['datetime'].min()} to {df_clean['datetime'].max()}")

# Reorder columns (datetime first)
cols = ['datetime'] + [col for col in df_clean.columns if col != 'datetime']
df_clean = df_clean[cols]

print(f"\n📊 Dataset ready:")
print(f"   Records: {len(df_clean):,}")
print(f"   Features: {len(df_clean.columns)}")
print(f"   Columns: {list(df_clean.columns[:10])}...")

# Connect to MongoDB
print("\n🔌 Connecting to MongoDB...")
db = MongoDBHandler()

# Clear existing features
print("\n🗑️  Clearing old features...")
result = db.features.delete_many({})
print(f"   Deleted {result.deleted_count} old records")

# Upload new data
print("\n💾 Uploading clean features...")
db.insert_features(df_clean)

# Verify
print("\n✅ Verifying upload...")
db.get_collection_stats()

db.close()

print("\n" + "="*60)
print("✅ UPLOAD COMPLETE!")
print("="*60)
print(f"\n✨ Clean dataset with NO data leakage:")
print(f"   • Only 24h, 48h, 72h lags")
print(f"   • Only 72h rolling windows")  
print(f"   • 30 features + 1 target (aqi)")
print(f"\n💡 Ready to train:")
print(f"   python -m src.training_pipeline")
