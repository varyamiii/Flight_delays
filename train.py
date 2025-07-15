# train.py
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_loader import load_data
from preprocess import encode_categoricals, extract_dep_hour
from feature_engineering import integrate_weather, create_sequences
from model import build_model
from config import *

def objective(trial):
    df = load_data()
    df = encode_categoricals(df, ['Origin','Dest','Reporting_Airline','OriginState','DestState'])
    df = extract_dep_hour(df)
    df = integrate_weather(df, GEOAPIFY_KEY)
    X = df[BASE_FEATURES].values
    y = df[TARGET].values
    X = StandardScaler().fit_transform(X)
    Xs, ys = create_sequences(X, y)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42)

    params = {
        "units1": trial.suggest_int(*SEARCH_SPACE["units1"]),
        "units2": trial.suggest_int(*SEARCH_SPACE["units2"]),
        "heads": trial.suggest_int(*SEARCH_SPACE["heads"]),
        "lr": trial.suggest_float(*SEARCH_SPACE["lr"], log=True),
        "bs": trial.suggest_categorical("bs", SEARCH_SPACE["bs"]),
        "epochs": trial.suggest_int(*SEARCH_SPACE["epochs"])
    }

    m = build_model(SEQ_LEN, Xtr.shape[2], params["units1"], params["units2"], params["heads"])
    m.optimizer.learning_rate = params["lr"]
    m.fit(Xtr, ytr, validation_data=(Xte,yte),
          epochs=params["epochs"], batch_size=params["bs"], verbose=0)
    pred = m.predict(Xte)
    return np.sqrt(mean_squared_error(yte, pred))

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print("Best params:", study.best_params)