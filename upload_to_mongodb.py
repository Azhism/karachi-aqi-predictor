"""
Upload cleaned_karachi_features.csv to MongoDB
Run this ONCE to initialize your database
"""
import pandas as pd
from src.database import MongoDBHandler

print("="*60)
print("📤 UPLOADING DATA TO MONGODB")
print("="*60)

# Load your CSV file
print("\n📂 Loading CSV file...")
try:
    # Try loading from data folder first
    df = pd.read_csv('data/karachi_aqi_direct_dataset.csv')
except FileNotFoundError:
    # If not found, try root folder
    try:
        df = pd.read_csv('karachi_aqi_direct_dataset.csv')
    except FileNotFoundError:
        print("❌ ERROR: CSV file not found!")
        print("   Please place 'karachi_aqi_direct_dataset.csv' in:")
        print("   1. data/ folder, OR")
        print("   2. Root folder")
        exit(1)

print(f"✅ Loaded dataset:")
print(f"   Records: {len(df):,}")
print(f"   Features: {len(df.columns)}")
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Convert datetime column
df['datetime'] = pd.to_datetime(df['datetime'])

# Connect to MongoDB
print("\n🔌 Connecting to MongoDB...")
try:
    db = MongoDBHandler()
except Exception as e:
    print(f"❌ ERROR connecting to MongoDB: {e}")
    print("\n⚠️  Please check:")
    print("   1. MongoDB URI in .env file is correct")
    print("   2. MongoDB Atlas cluster is running")
    print("   3. Network access is configured (0.0.0.0/0)")
    print("   4. Database user credentials are correct")
    exit(1)

# Upload to features collection
print("\n💾 Uploading to MongoDB...")
db.insert_features(df)

# Verify upload
print("\n✅ Verifying upload...")
db.get_collection_stats()

# Close connection
db.close()

print("\n" + "="*60)
print("✅ UPLOAD COMPLETE!")
print("="*60)
print("\nYour data is now in MongoDB and ready for:")
print("   • Feature pipeline")
print("   • Model training")
print("   • Predictions")
