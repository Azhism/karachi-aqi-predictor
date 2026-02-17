from src.database import MongoDBHandler

db = MongoDBHandler()

print("🗑️  CLEARING FEATURES COLLECTION...")
print(f"Current records: {db.features.count_documents({})}")

# Delete all features
result = db.features.delete_many({})
print(f"✅ Deleted {result.deleted_count} records")

print(f"\nNew count: {db.features.count_documents({})}")
print("\n✅ Ready for fresh hourly data collection!")
print("   Next GitHub Actions run will start collecting real-time data.")
