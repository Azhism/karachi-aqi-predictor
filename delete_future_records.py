"""
Delete future records from MongoDB to fix hourly data collection

This script removes records with timestamps in the future, allowing the
hourly pipeline to resume adding new real-time data normally.
"""
from src.database import MongoDBHandler
from datetime import datetime

def delete_future_records():
    db = MongoDBHandler()
    
    # Get current UTC time
    now_utc = datetime.utcnow()
    print("="*70)
    print("🔍 FUTURE RECORDS CLEANUP")
    print("="*70)
    print(f"\nCurrent UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count future records
    future_count = db.features.count_documents({"datetime": {"$gt": now_utc}})
    total_count = db.features.count_documents({})
    
    print(f"\nTotal records in database: {total_count}")
    print(f"Future records (beyond current time): {future_count}")
    
    if future_count == 0:
        print("\n✅ No future records found. Your database is clean!")
        return
    
    # Show the range that will be deleted
    earliest_future = db.features.find_one(
        {"datetime": {"$gt": now_utc}},
        sort=[("datetime", 1)]
    )
    latest_future = db.features.find_one(
        {"datetime": {"$gt": now_utc}},
        sort=[("datetime", -1)]
    )
    
    print(f"\n📅 Future records range:")
    print(f"   From: {earliest_future['datetime']}")
    print(f"   To:   {latest_future['datetime']}")
    
    # Show what latest record will be after deletion
    latest_real = db.features.find_one(
        {"datetime": {"$lte": now_utc}},
        sort=[("datetime", -1)]
    )
    
    if latest_real:
        print(f"\n📌 After deletion, latest record will be:")
        print(f"   {latest_real['datetime']} - AQI: {latest_real.get('aqi', 'N/A')}")
    
    # Confirm before proceeding
    print(f"\n⚠️  This will delete {future_count} records with timestamps in the future.")
    print("   This is SAFE - you're removing impossible timestamps.")
    print("   Your hourly pipeline will then resume adding real data normally.")
    
    confirm = input(f"\nDelete {future_count} future records? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("\n❌ Aborted. No records deleted.")
        return
    
    # Delete future records
    print(f"\n🗑️  Deleting future records...")
    result = db.features.delete_many({"datetime": {"$gt": now_utc}})
    
    print(f"\n✅ Successfully deleted {result.deleted_count} future records!")
    
    # Show final state
    new_total = db.features.count_documents({})
    latest_now = db.features.find_one(sort=[("datetime", -1)])
    
    print(f"\n📊 Database status:")
    print(f"   Total records: {new_total}")
    print(f"   Latest timestamp: {latest_now['datetime']}")
    print(f"   Latest AQI: {latest_now.get('aqi', 'N/A')}")
    
    print(f"\n🎉 Done! Your hourly GitHub Actions will now add new records starting from:")
    print(f"   {latest_now['datetime']}")

if __name__ == "__main__":
    delete_future_records()
