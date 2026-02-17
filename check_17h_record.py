"""
To check if the hourly workflow ran on GitHub Actions:

1. Go to: https://github.com/Azhism/karachi-aqi-predictor/actions

2. Look for "Hourly Data Collection" workflow

3. Check if there's a run at ~12:00 UTC (5:00 PM local)

4. If it ran successfully, check the logs to see if data was added

Possible reasons if no new data:
- Workflow hasn't triggered yet (cron can have a few minutes delay)
- OpenWeather API returned same timestamp
- Workflow failed
- Data already exists (duplicate check prevented insertion)
"""

from src.database import MongoDBHandler
from datetime import datetime, timedelta

db = MongoDBHandler()

print("🔍 CHECKING FOR NEW DATA (17:00 record)\n")

# Check for 17:00 record specifically
target_time = datetime(2026, 2, 17, 17, 0, 0)
record_17 = db.features.find_one({'datetime': target_time})

if record_17:
    print(f"✅ Found 17:00 (5:00 PM) record!")
    print(f"   AQI: {record_17.get('aqi')}")
    print(f"   Temperature: {record_17.get('temperature')}°C")
else:
    print(f"❌ No 17:00 (5:00 PM) record found yet")
    print(f"\nPossible reasons:")
    print(f"   • Workflow hasn't run yet (cron can delay)")
    print(f"   • Workflow is still running (takes ~3-5 minutes)")
    print(f"   • OpenWeather API hasn't updated to 17:00 yet")
    print(f"\n💡 Check GitHub Actions: https://github.com/Azhism/karachi-aqi-predictor/actions")

# Show last 5 records
print(f"\n📊 Last 5 records in database:")
last_5 = list(db.features.find().sort('datetime', -1).limit(5))
for rec in last_5:
    print(f"   {rec['datetime']} - AQI: {rec.get('aqi')}")

db.close()
