import pandas as pd

df = pd.read_csv('data/karachi_aqi_direct_dataset.csv')

print('AQI vs PM2.5 relationship:')
for aqi in sorted(df['aqi'].unique()):
    subset = df[df['aqi'] == aqi]
    print(f'AQI {aqi}: PM2.5 range = {subset["pm2_5"].min():.1f} - {subset["pm2_5"].max():.1f}')

print('\nChecking if dataset has european_aqi or any other AQI column:')
aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
print('AQI-related columns:', aqi_cols)
