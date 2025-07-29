import pickle
import os

MODEL_PATH = os.path.join("models", "is_intrusion_model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_anomalies(model, X, original_df):
    preds = model.predict(X)  # -1 or 1 labels
    scores = model.decision_function(X)  # continuous anomaly scores

    original_df["anomaly_label"] = preds  # renamed for clarity
    original_df["anomaly_score"] = scores  # the real continuous score

    for idx, row in original_df.iterrows():
        print(f"Entry {idx}: user={row['user']}, ip={row['ip_address']}, label={row['anomaly_label']}, score={row['anomaly_score']:.4f}")

    anomalies = original_df[original_df["anomaly_label"] == -1]

    summary = {
        "total_entries": len(original_df),
        "anomalies_found": len(anomalies)
    }

    return {
        "summary": summary,
        "anomalies": anomalies.to_dict(orient="records")
    }


