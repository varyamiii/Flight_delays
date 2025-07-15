# pipeline.py
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from config import *
from data_loader import load_data
from preprocess import encode_categoricals, extract_dep_hour
from feature_engineering import integrate_weather, create_sequences
from model import build_model
from evaluate import evaluate


class FlightDelayPipeline:
    def __init__(self, seq_len=SEQ_LEN, geo_key=GEOAPIFY_KEY):
        self.seq_len = seq_len
        self.geo_key = geo_key
        self.scaler = StandardScaler()
        self.best_params = None
        self.model = None

    def prepare_data(self):
        df = load_data()
        df = encode_categoricals(df, ['Origin', 'Dest', 'Reporting_Airline', 'OriginState', 'DestState'])
        df = extract_dep_hour(df)
        df = integrate_weather(df, self.geo_key)
        df = df.dropna()

        X = df[BASE_FEATURES].values
        y = df[TARGET].values

        X = self.scaler.fit_transform(X)
        self.X_seq, self.y_seq = create_sequences(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_seq, self.y_seq, test_size=0.2, random_state=42
        )

    def _objective(self, trial):
        params = {
            "units1": trial.suggest_int(*SEARCH_SPACE["units1"]),
            "units2": trial.suggest_int(*SEARCH_SPACE["units2"]),
            "heads": trial.suggest_int(*SEARCH_SPACE["heads"]),
            "lr": trial.suggest_float(*SEARCH_SPACE["lr"], log=True),
            "bs": trial.suggest_categorical("bs", SEARCH_SPACE["bs"]),
            "epochs": trial.suggest_int(*SEARCH_SPACE["epochs"])
        }

        model = build_model(self.seq_len, self.X_train.shape[2],
                            params["units1"], params["units2"], params["heads"])
        model.optimizer.learning_rate = params["lr"]
        model.fit(self.X_train, self.y_train,
                  validation_data=(self.X_test, self.y_test),
                  epochs=params["epochs"], batch_size=params["bs"],
                  verbose=0)
        preds = model.predict(self.X_test)
        return np.sqrt(mean_squared_error(self.y_test, preds))

    def optimize(self, n_trials=OPTUNA_TRIALS):
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)
        self.best_params = study.best_params
        print("Best hyperparameters found:", self.best_params)

    def train_final_model(self):
        if self.best_params is None:
            raise ValueError("You must run optimize() before training final model.")

        p = self.best_params
        self.model = build_model(self.seq_len, self.X_train.shape[2],
                                 p["units1"], p["units2"], p["heads"])
        self.model.optimizer.learning_rate = p["lr"]
        self.model.fit(self.X_train, self.y_train,
                       validation_data=(self.X_test, self.y_test),
                       epochs=p["epochs"], batch_size=p["bs"],
                       verbose=1)

    def evaluate_model(self):
        if self.model is None:
            raise ValueError("Model is not trained. Call train_final_model() first.")
        preds = self.model.predict(self.X_test)
        evaluate(self.y_test, preds)