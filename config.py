# config.py
CSV_PATH = "data/flight_delay_predict.csv"

GEOAPIFY_KEY = "eab0531ec1de4bf0821cb10eb35d9952"

SEQ_LEN = 10

OPTUNA_TRIALS = 20

SEARCH_SPACE = {
    "units1": (32, 128),
    "units2": (16, 64),
    "heads": (1, 4),
    "lr": (1e-4, 1e-2),
    "bs": [16, 32, 64],
    "epochs": (10, 50)
}

BASE_FEATURES = [
    'Year','Month','DayOfWeek','DepHour',
    'Distance','AirTime','Origin','Dest',
    'Reporting_Airline','temp','precip','wind'
]
TARGET = 'ArrDelayMinutes'
