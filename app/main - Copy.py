from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import logging
from typing import List, Dict, Any
import pandas as pd
import re
from fastapi.encoders import jsonable_encoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Auth Log Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuthLogAnalyzer:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained IsolationForest model"""
        try:
            model_path = os.path.join("models", "is_intrusion_model.pkl")
            if not os.path.exists(model_path):
                logger.warning("IsolationForest model not found.")
                return
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("IsolationForest model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """Parse a single auth log line"""
        pattern = r'(?P<timestamp>\w{3} \d+ \d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<service>\S+): (?P<message>.+)'
        match = re.match(pattern, line.strip())
        if not match:
            return None

        message = match.group("message")
        user_match = re.search(r'(?:for|user) (?:invalid user )?(\w+)', message)
        user = user_match.group(1) if user_match else "unknown"
        ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', message)
        ip_address = ip_match.group(1) if ip_match else "0.0.0.0"
        success = 'Accepted' in message

        return {
            "raw_line": line.strip(),
            "timestamp": match.group("timestamp"),
            "user": user,
            "ip_address": ip_address,
            "success": success,
            "event_type": "login_attempt"
        }

    def extract_features(self, entries: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract numerical features for anomaly detection"""
        df = pd.DataFrame(entries)
        df["success"] = df["success"].astype(int)
        df["ip_address"] = df["ip_address"].astype(str)
        df["user"] = df["user"].astype(str)

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

        expected_cols = ["user_success_rate", "ip_login_count", "is_internal_ip", "user_login_count"]
        return df[expected_cols]

    def analyze_log_lines(self, log_lines: List[str]) -> Dict[str, Any]:
        parsed_logs = [self.parse_log_line(line) for line in log_lines]
        parsed_logs = [log for log in parsed_logs if log]

        if not parsed_logs:
            raise ValueError("No valid log entries found.")

        features_df = self.extract_features(parsed_logs)
        scores = self.model.decision_function(features_df)
        predictions = self.model.predict(features_df)

        results = []
        threshold_high = -0.2
        threshold_medium = 0.0
        for entry, score, pred in zip(parsed_logs, scores, predictions):
            anomaly_score = float(-score)
            is_anomaly = bool(pred == -1)
            #severity = "high" if anomaly_score >= 0.8 else "medium" if anomaly_score >= 0.6 else "low"
            severity = (
                "high" if score < threshold_high
                else "medium" if score < threshold_medium
                else "low"
                        )

            results.append({
                'timestamp': entry.get('timestamp'),
                'event_type': entry.get('event_type'),
                'user': entry.get('user'),
                'ip_address': entry.get('ip_address'),
                'success': int(entry.get('success', 0)),
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'anomaly_severity': severity
            })

        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        high_severity_count = sum(1 for r in results if r['anomaly_severity'] == 'high')

        return jsonable_encoder({
            'summary': {
                'total_entries': len(results),
                'anomalies_detected': anomaly_count,
                'anomaly_percentage': (anomaly_count / len(results)) * 100 if results else 0.0,
                'high_severity_anomalies': high_severity_count
            },
            'anomalies': [r for r in results if r['is_anomaly']],
            'all_entries': results
        })

# Initialize analyzer
analyzer = AuthLogAnalyzer()

@app.post("/analyze")
async def analyze_log(file: UploadFile = File(...)):
    try:
        content = await file.read()
        log_lines = content.decode("utf-8").splitlines()
        log_lines = [line.strip() for line in log_lines if line.strip()]
        result = analyzer.analyze_log_lines(log_lines)
        return result
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {"error": f"Error processing file: {str(e)}"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": analyzer.model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
