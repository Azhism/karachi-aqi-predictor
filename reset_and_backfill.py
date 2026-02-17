"""
Reset database and populate with fresh real-time data from OpenWeather API
"""
from src.feature_pipeline import FeaturePipeline
from src.database import MongoDBHandler

print("="*70)
print("🔄 DATABASE RESET & BACKFILL")
print("="*70)

# Step 1: Clear old data
db = MongoDBHandler()
old_count = db.features.count_documents({})
print(f"\n🗑️  Removing {old_count} old records (training dataset)...")
db.features.delete_many({})
print("✅ Database cleared!")

# Step 2: Backfill with real data
print("\n📥 Fetching last 2 days of real-time data from OpenWeather...")
fp = FeaturePipeline()
success = fp.run_backfill(days=2)

if success:
    new_count = db.features.count_documents({})
    print(f"\n✅ SUCCESS! Database now has {new_count} real-time records")
    print("🔄 Hourly automation will add new records from now on!")
else:
    print("\n❌ Backfill failed. Check error messages above.")
