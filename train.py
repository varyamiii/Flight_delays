# train.py
from pipeline import FlightDelayPipeline

if __name__ == "__main__":
    pipe = FlightDelayPipeline()
    pipe.prepare_data()
    pipe.optimize()
    pipe.train_final_model()
    pipe.evaluate_model()
