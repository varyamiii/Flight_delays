import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import optuna

# Загрузка данных
df_base = pd.read_csv("the best.csv", delimiter=',')

# Удаление ненужных столбцов
df = df_base.drop(["Cancelled", "Diverted", "DayofMonth", "DistanceGroup"], axis=1)

# Преобразование категориальных признаков в числовые
df['Origin'] = LabelEncoder().fit_transform(df['Origin'])
df['Dest'] = LabelEncoder().fit_transform(df['Dest'])
df['Reporting_Airline'] = LabelEncoder().fit_transform(df['Reporting_Airline'])
df['OriginState'] = LabelEncoder().fit_transform(df['OriginState'])
df['DestState'] = LabelEncoder().fit_transform(df['DestState'])


# Преобразование времени вылета
def convert_CRSDepTime(time):
    time_str = str(time).zfill(4)
    hours = time_str[:2]
    minutes = time_str[2:]
    standard_time = f"{hours}:{minutes}:00"
    return standard_time


df['CRSDepTime'] = df['CRSDepTime'].apply(convert_CRSDepTime)
df['DepHour'] = df['CRSDepTime'].str[:2].astype(int)

# Выбор признаков и целевой переменной
features = ['Year', 'Month', 'DayOfWeek', 'DepHour', 'Distance', 'AirTime', 'Origin', 'Dest', 'Reporting_Airline']
target = 'ArrDelayMinutes'

X = df[features]
y = df[target]

# Масштабирование числовых признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Группировка данных по Origin и Reporting_Airline
grouped = df.groupby(['Origin', 'Reporting_Airline'])


# Функция для создания последовательностей
def create_sequences(group, seq_length):
    xs, ys = [], []
    for i in range(len(group) - seq_length):
        x = group[i:i + seq_length]
        y = group['ArrDelayMinutes'].values[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 10  # Длина последовательности
X_seq, y_seq = [], []

for (_, _), group in grouped:
    # Сортировка данных по времени вылета
    group = group.sort_values(by=['FlightDate', 'CRSDepTime'])

    # Создание последовательностей
    X_group_seq, y_group_seq = create_sequences(group[features].values, seq_length)
    X_seq.extend(X_group_seq)
    y_seq.extend(y_group_seq)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Разделение данных на обучающую и тестовую выборки
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)


# Создание модели LSTM
def create_lstm_model(units=50, learning_rate=0.001):
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(seq_length, X_train_seq.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Оптимизация гиперпараметров с помощью Optuna
def objective(trial):
    # Гиперпараметры для оптимизации
    units = trial.suggest_int('units', 32, 128)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Создание и обучение модели
    model = create_lstm_model(units=units)
    model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs, batch_size=batch_size, verbose=0
    )

    # Оценка модели
    y_pred = model.predict(X_test_seq)
    rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
    return rmse


# Запуск оптимизации
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Лучшие параметры
print("Best parameters:", study.best_params)

# Обучение модели с лучшими параметрами
best_units = study.best_params['units']
best_learning_rate = study.best_params['learning_rate']
best_batch_size = study.best_params['batch_size']
best_epochs = study.best_params['epochs']

best_model = create_lstm_model(units=best_units)
best_model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=best_epochs, batch_size=best_batch_size, verbose=1
)

# Оценка лучшей модели
y_pred_best = best_model.predict(X_test_seq)
rmse_best = np.sqrt(mean_squared_error(y_test_seq, y_pred_best))
print("Best RMSE:", rmse_best)
