"""Delete all features from MongoDB to regenerate with clean config"""
from src.database import MongoDBHandler

db = MongoDBHandler()

print("🗑️  Deleting all features from MongoDB...")
result = db.features.delete_many({})
print(f"✅ Deleted {result.deleted_count} feature records")
print("\n💡 Now run: python -m src.feature_pipeline")
print("   This will regenerate features WITHOUT data leakage")

db.close()
