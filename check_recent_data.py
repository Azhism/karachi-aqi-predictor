"""
Quick script to check when the last data was added to MongoDB
"""
from src.database import MongoDBHandler
from datetime import datetime

print("🔍 Checking MongoDB for recent data...\n")

db = MongoDBHandler()

# Get latest records
latest = db.get_latest_features(n_hours=24)

if latest is not None and len(latest) > 0:
    print(f"✅ Found {len(latest)} records from last 24 hours\n")
    print("📅 5 Most Recent Records:")
    for i, record in enumerate(latest.tail(5).to_dict('records'), 1):
        dt = record.get('datetime', 'Unknown')
        print(f"{i}. {dt}")
        if i == 1:
            # Calculate time since last update
            if isinstance(dt, str):
                from dateutil import parser
                dt = parser.parse(dt)
            time_diff = datetime.now(dt.tzinfo if dt.tzinfo else None) - dt
            hours_ago = time_diff.total_seconds() / 3600
            print(f"   ⏰ Last update was {hours_ago:.1f} hours ago")
    
    # Get stats
    stats = db.get_collection_stats()
    print(f"\n📊 Database Statistics:")
    print(f"   Total features: {stats.get('features', {}).get('count', 0):,}")
else:
    print("❌ No data found in database")

print("\n" + "="*60)
print("If the last record is older than 1-2 hours,")
print("check GitHub Actions workflow status!")
print("="*60)
