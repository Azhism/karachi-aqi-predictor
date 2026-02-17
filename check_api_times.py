import requests
from datetime import datetime, timedelta

# Check what OpenWeather API returns
end = datetime.utcnow()
start = end - timedelta(hours=12)

from src.config import OPENWEATHER_API_KEY

url = 'http://api.openweathermap.org/data/2.5/air_pollution/history'
params = {
    'lat': 24.8608,
    'lon': 67.0104,
    'start': int(start.timestamp()),
    'end': int(end.timestamp()),
    'appid': OPENWEATHER_API_KEY
}

print(f"Current UTC time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Requesting data from: {start.strftime('%Y-%m-%d %H:%M:%S')} to {end.strftime('%Y-%m-%d %H:%M:%S')}")

r = requests.get(url, params=params)
data = r.json()

print(f"\nTotal records in API response: {len(data.get('list', []))}")
print("\nLast 10 timestamps from OpenWeather API:")

for record in data.get('list', [])[-10:]:
    dt = datetime.utcfromtimestamp(record['dt'])
    aqi = record['main']['aqi']
    pm25 = record['components']['pm2_5']
    print(f"  {dt.strftime('%Y-%m-%d %H:%M')} UTC - AQI: {aqi}, PM2.5: {pm25:.2f}")
