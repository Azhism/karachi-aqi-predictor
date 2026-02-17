"""Check new 30-column dataset"""
import pandas as pd

df = pd.read_csv('data/karachi_aqi_30features.csv')

print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nData types:")
print(df.dtypes)

print(f"\nFirst 3 rows:")
print(df.head(3))

if 'datetime' in df.columns:
    print(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total records: {len(df):,}")
