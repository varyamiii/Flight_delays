# data_loader.py
import pandas as pd
from config import CSV_PATH

def load_data():
    return pd.read_csv(CSV_PATH, parse_dates=["FlightDate"])