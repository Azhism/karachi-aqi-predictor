"""Remove dropped columns from MongoDB"""
from src.database import MongoDBHandler

db = MongoDBHandler()

# Columns that were dropped during EDA
dropped_cols = ['precipitation', 'aqi_change_1h', 'aqi_change_3h']

print("🗑️  Removing dropped columns from MongoDB features collection...\n")

# Remove these fields from all documents
result = db.features.update_many(
    {},
    {'$unset': {col: "" for col in dropped_cols}}
)

print(f"✅ Updated {result.modified_count} documents")
print(f"   Removed columns: {', '.join(dropped_cols)}")

# Verify
sample = db.features.find_one()
if sample:
    remaining_cols = [col for col in dropped_cols if col in sample]
    if remaining_cols:
        print(f"\n⚠️  Still found: {remaining_cols}")
    else:
        print(f"\n✅ All dropped columns successfully removed!")
    print(f"   Total columns now: {len(sample) - 1}")  # -1 for _id

db.close()
