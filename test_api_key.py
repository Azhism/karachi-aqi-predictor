"""
Test if OpenWeather API key works in feature pipeline
"""
import os
from src.config import OPENWEATHER_API_KEY

print("="*70)
print("🔑 API KEY TEST")
print("="*70)

print(f"\nOpenWeather API Key from config:")
print(f"   Value: {OPENWEATHER_API_KEY[:10]}...{OPENWEATHER_API_KEY[-5:] if OPENWEATHER_API_KEY else 'MISSING'}")
print(f"   Length: {len(OPENWEATHER_API_KEY) if OPENWEATHER_API_KEY else 0} chars")
print(f"   Is set: {'✅ YES' if OPENWEATHER_API_KEY else '❌ NO'}")

# Test actual API call
if OPENWEATHER_API_KEY:
    print("\n🌐 Testing OpenWeather API call...")
    import requests
    from datetime import datetime, timedelta
    
    end = datetime.now()
    start = end - timedelta(hours=2)
    
    url = 'http://api.openweathermap.org/data/2.5/air_pollution/history'
    params = {
        'lat': 24.8608,
        'lon': 67.0104,
        'start': int(start.timestamp()),
        'end': int(end.timestamp()),
        'appid': OPENWEATHER_API_KEY
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        print(f"   Status code: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"   ✅ API call successful!")
            print(f"   Records received: {len(data.get('list', []))}")
        else:
            print(f"   ❌ API error: {r.text}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
else:
    print("\n❌ Cannot test - API key not set!")
