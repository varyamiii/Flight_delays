# Flight Delays Prediction

Прогнозирование задержек авиарейсов с учетом погодных условий, временных и географических признаков с использованием нейросетевой архитектуры LSTM + Multi-Head Attention и оптимизации гиперпараметров через Optuna.

## Описание проекта

Проект направлен на построение модели, предсказывающей количество минут задержки прибытия авиарейсов. Особенность подхода — интеграция погодных данных по координатам аэропортов, полученных через внешние API (Geoapify + Open-Meteo), и обучение модели на временных последовательностях с помощью рекуррентной нейросети.

## Стек технологий

- Python 3.10+
- Pandas, NumPy, scikit-learn — обработка данных, метрики
- TensorFlow / Keras — LSTM + Multi-Head Attention модель
- Optuna — автоматический подбор гиперпараметров
- Geoapify API — геокодирование IATA-кодов
- Open-Meteo API — погодные данные по координатам

## Архитектура проекта
```bash
Flight_delays/
├── config.py # Глобальные параметры и пути
├── data_loader.py # Загрузка данных
├── preprocess.py # Категориальные признаки, DepHour
├── feature_engineering.py # Погодные данные, создание последовательностей
├── model.py # Архитектура LSTM + Attention
├── pipeline.py # Класс FlightDelayPipeline (вся логика проекта)
├── train.py # Основной скрипт запуска пайплайна
├── evaluate.py # Метрики модели
├── utils.py # Работа с внешними API (Geoapify, Open-Meteo)
├── requirements.txt # Все зависимости проекта
└── data/
   └── flight_delay_predict.csv # CSV-файл с данными о рейсах
```

## Как запустить

1. **Клонируй репозиторий:**
   ```bash
   git clone https://github.com/varyamiii/Flight_delays.git
   cd Flight_delays

2. **Создай виртуальное окружение и установи зависимости:**
   ```bash
   python -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate
   pip install -r requirements.txt


3. **Добавь файл с данными:**
    ```bash
    data/flight_delay_predict.csv


4. **Запусти обучение модели:**
    ```bash
    python train.py


Модель автоматически выполнит:
- Предобработку данных
- Парсинг координат аэропортов и загрузку погодных условий
- Обучение LSTM Attention модели с Optuna
- Оценку финального результата

## Выходные метрики

- RMSE
- MAE
- R²

Метрики выводятся после обучения модели.

## Примечание

Для работы с Geoapify необходимо указать свой API-ключ в `config.py`: 
```bash
GEOAPIFY_KEY = "ваш_ключ"
```
Получить ключ можно на сайте: https://www.geoapify.com/
