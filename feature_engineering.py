# feature_engineering.py
import numpy as np, pandas as pd
from config import BASE_FEATURES, TARGET, SEQ_LEN
from utils import geocode_iata, fetch_weather
import time

def integrate_weather(df, geo_key):
    cache = {}
    cols = ['temp','precip','wind']
    df[cols] = np.nan
    for i, row in df.iterrows():
        key = row['Origin']
        if key not in cache:
            cache[key] = geocode_iata(row['Origin'], geo_key)
        lat, lon = cache[key]
        w = fetch_weather(lat, lon, row['FlightDate'].strftime("%Y-%m-%d"))
        hour_df = pd.DataFrame(w).T
        try:
            rec = hour_df[hour_df['time'].str.contains(f"T{row['DepHour']:02d}:")].iloc[0]
            df.loc[i, cols] = rec[['temperature_2m','precipitation','wind_speed_10m']].values
        except:
            df.loc[i, cols] = [np.nan]*3
        time.sleep(0.1)
    return df.dropna()

def create_sequences(X, y):
    xs, ys = [], []
    for i in range(len(X)-SEQ_LEN):
        xs.append(X[i:i+SEQ_LEN])
        ys.append(y[i+SEQ_LEN])
    return np.array(xs), np.array(ys)
