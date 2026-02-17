from src.database import MongoDBHandler
from datetime import datetime, timedelta
import pandas as pd
import requests

db = MongoDBHandler()

print("="*70)
print("🔍 TIMESTAMP INVESTIGATION")
print("="*70)

# Get latest DB record
latest_docs = list(db.features.find().sort('datetime', -1).limit(3))
print("\n📊 Latest 3 DB records:")
for doc in latest_docs:
    print(f"   {doc['datetime']} - AQI: {doc['aqi']}")

latest_db = latest_docs[0]['datetime']

# Current time
now_utc = datetime.utcnow()
now_local = datetime.now()

print(f"\n🕐 Current time:")
print(f"   Local: {now_local.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   UTC:   {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n📅 Latest DB record: {latest_db}")
print(f"   Time difference from now (UTC): {now_utc - latest_db}")

# Check what OpenWeather API returns now
print("\n🌐 Checking OpenWeather API...")
end = datetime.now()
start = end - timedelta(hours=6)

from src.config import OPENWEATHER_API_KEY

url = 'http://api.openweathermap.org/data/2.5/air_pollution/history'
params = {
    'lat': 24.8608,
    'lon': 67.0104,
    'start': int(start.timestamp()),
    'end': int(end.timestamp()),
    'appid': OPENWEATHER_API_KEY
}

r = requests.get(url, params=params)
data = r.json()

print(f"\n📡 OpenWeather returned {len(data.get('list', []))} records")
if len(data.get('list', [])) > 0:
    print("\nLast 3 timestamps from API:")
    for record in data['list'][-3:]:
        dt = datetime.fromtimestamp(record['dt'])
        print(f"   {dt.strftime('%Y-%m-%d %H:%M:%S')} - AQI: {record['main']['aqi']}")
    
    api_latest = datetime.fromtimestamp(data['list'][-1]['dt'])
    print(f"\n⏰ API latest timestamp: {api_latest}")
    print(f"   Difference from DB latest: {api_latest - latest_db}")
    print(f"   Is API newer than DB? {api_latest > latest_db}")
