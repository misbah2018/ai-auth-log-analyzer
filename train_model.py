import os
import pickle
import pandas as pd
from sklearn.ensemble import IsolationForest
from app.utils import parse_log_file, preprocess_df

LOG_PATH = "app/sample_auth.log"   # Or your full training log dataset path
MODEL_PATH = os.path.join("models", "is_intrusion_model.pkl")

def train_and_save_model(log_path=LOG_PATH, model_path=MODEL_PATH, contamination=0.1):
    print("Loading log data...")
    df = parse_log_file(log_path)

    if df.empty:
        raise ValueError("No log data found for training!")

    print("Preprocessing data...")
    X, _ = preprocess_df(df)

    print(f"Training IsolationForest with contamination={contamination}...")
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
