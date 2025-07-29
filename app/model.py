import pickle
import os

MODEL_PATH = os.path.join("models", "is_intrusion_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_anomalies(model, X, original_df):
    preds = model.predict(X)
    scores = model.decision_function(X)

    original_df["anomaly_score"] = preds
    original_df["decision_score"] = scores

    anomalies = original_df[original_df["anomaly_score"] == -1]

    # ✅ Convert float32 to Python float (to avoid JSON serialization issues)
    anomalies = anomalies.copy()
    anomalies["decision_score"] = anomalies["decision_score"].astype(float)

    summary = {
        "total_entries": len(original_df),
        "anomalies_found": len(anomalies)
    }

    return {
        "summary": summary,
        "anomalies": anomalies.to_dict(orient="records")
    }



