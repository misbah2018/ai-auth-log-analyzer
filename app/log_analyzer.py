import pandas as pd
import numpy as np
from typing import List, Dict, Any


class AuthLogAnalyzer:
    def __init__(self, anomaly_model, feature_extractor=None, label_encoders=None):
        self.anomaly_model = anomaly_model
        self.feature_extractor = feature_extractor
        self.label_encoders = label_encoders or {}

    def detect_intrusions(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Extract engineered features
            features_df = self.extract_features(entries)

            # Predict anomaly scores
            anomaly_scores = self.anomaly_model.decision_function(features_df)
            preds = self.anomaly_model.predict(features_df)  # -1 = anomaly, 1 = normal

            results = []
            for entry, score, pred in zip(entries, anomaly_scores, preds):
                results.append({
                    "entry": entry,
                    "anomaly_score": round(score, 4),
                    "is_anomaly": pred == -1
                })

            summary = {
                "total": len(results),
                "anomalies": sum(r["is_anomaly"] for r in results),
                "normal": sum(not r["is_anomaly"] for r in results),
            }

            return {"results": results, "summary": summary}

        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}

    def extract_features(self, entries: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(entries)

        df["ip_address"] = df["ip_address"].astype(str)
        df["user"] = df["user"].astype(str)
        df["success"] = df["success"].astype(int)

        ip_counts = df["ip_address"].value_counts().to_dict()
        user_counts = df["user"].value_counts().to_dict()
        success_map = df.groupby("user")["success"].mean().to_dict()

        def is_internal(ip):
            return ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172.")

        df["ip_login_count"] = df["ip_address"].map(ip_counts)
        df["user_login_count"] = df["user"].map(user_counts)
        df["user_success_rate"] = df["user"].map(success_map)
        df["is_internal_ip"] = df["ip_address"].apply(lambda ip: int(is_internal(ip)))

        df.fillna(0, inplace=True)

        # ✅ Must match training column order exactly
        expected_cols = ["user_success_rate", "ip_login_count", "is_internal_ip", "user_login_count"]
        return df[expected_cols]
