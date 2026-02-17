"""
Check all data in MongoDB to see what was added
"""
from src.database import MongoDBHandler
from datetime import datetime

print("🔍 Checking ALL data in MongoDB...\n")

db = MongoDBHandler()

# Get ALL features
all_features = db.get_features(limit=None)

if all_features is not None and len(all_features) > 0:
    print(f"📊 Total records: {len(all_features):,}\n")
    
    # Sort by datetime descending
    all_features = all_features.sort_values('datetime', ascending=False)
    
    print("📅 10 Most Recent Records:")
    for i, record in enumerate(all_features.head(10).to_dict('records'), 1):
        dt = record.get('datetime', 'Unknown')
        print(f"{i}. {dt}")
    
    # Date range
    print(f"\n📆 Date Range:")
    print(f"   Oldest: {all_features['datetime'].min()}")
    print(f"   Newest: {all_features['datetime'].max()}")
    
    # Check if there's data from today
    today = datetime.now().date()
    today_data = all_features[all_features['datetime'].dt.date == today]
    print(f"\n🗓️  Records from today ({today}): {len(today_data)}")
    
else:
    print("❌ No data found in database")
