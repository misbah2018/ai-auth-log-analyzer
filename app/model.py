import pickle
import os

MODEL_PATH = os.path.join("models", "is_intrusion_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_anomalies(model, X, original_df):
    preds = model.predict(X)
    original_df["anomaly_score"] = preds
    anomalies = original_df[original_df["anomaly_score"] == -1]

    summary = {
        "total_entries": len(original_df),
        "anomalies_found": len(anomalies)
    }

    return {
        "summary": summary,
        "anomalies": anomalies.to_dict(orient="records")
    }
