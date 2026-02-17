from src.database import MongoDBHandler
from datetime import datetime

db = MongoDBHandler()

# Get latest record
latest = list(db.features.find().sort('datetime', -1).limit(1))[0]
print(f"Latest DB timestamp: {latest['datetime']}")
print(f"Type: {type(latest['datetime'])}")

# Get oldest record
oldest = list(db.features.find().sort('datetime', 1).limit(1))[0]
print(f"\nOldest DB timestamp: {oldest['datetime']}")

# Current time
print(f"\nCurrent UTC time: {datetime.utcnow()}")
print(f"Current local time: {datetime.now()}")

# Count total
total = db.features.count_documents({})
print(f"\nTotal records: {total}")
