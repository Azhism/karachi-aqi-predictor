import pandas as pd

# Read the CSV
df = pd.read_csv('data/karachi_aqi_direct_dataset.csv')

print("📂 CSV Dataset Info:")
print(f"   Total records: {len(df)}")
print(f"   First timestamp: {df['datetime'].iloc[0]}")
print(f"   Last timestamp:  {df['datetime'].iloc[-1]}")

print("\n📊 Latest 5 timestamps from CSV:")
print(df['datetime'].tail(5).values)
