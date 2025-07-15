# utils.py
import requests

def geocode_iata(iata: str, api_key: str):
    url = f"https://api.geoapify.com/v1/geocode/search?text={iata}&type=airport&apiKey={api_key}"
    res = requests.get(url).json()
    if res.get('features'):
        prop = res['features'][0]['properties']
        return prop['lat'], prop['lon']
    raise ValueError(f"Не удалось найти координаты {iata}")

def fetch_weather(lat, lon, date):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
    )
    df = requests.get(url).json()['hourly']
    return df