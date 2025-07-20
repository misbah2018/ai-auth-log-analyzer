import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class AuthLogAnalyzer:
    def __init__(self, anomaly_model, feature_extractor, label_encoder):
        self.anomaly_model = anomaly_model
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder

    def parse_log_line(self, line: str) -> Dict[str, Any]:
        # Minimal parser for demo (customize as needed)
        pattern = r'(?P<timestamp>\w{3} \d+ \d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<service>\S+): (?P<message>.+)'
        match = re.match(pattern, line)
        if not match:
            return None

        message = match.group("message")
        user_match = re.search(r'user (\w+)', message)
        ip_match = re.search(r'from (\d+\.\d+\.\d+\.\d+)', message)
        success = 'Accepted' in message

        return {
            "raw_line": line,
            "timestamp": match.group("timestamp"),
            "user": user_match.group(1) if user_match else "unknown",
            "ip_address": ip_match.group(1) if ip_match else "0.0.0.0",
            "success": success,
            "event_type": "login_attempt"
        }

    def extract_features(self, entries: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(entries)
        df['success'] = df['success'].astype(int)

        if self.label_encoder:
            try:
                df['user'] = self.label_encoder.transform(df['user'])
                df['ip_address'] = self.label_encoder.transform(df['ip_address'])
            except Exception as e:
                df['user'] = df['user'].apply(
                    lambda x: self.label_encoder.transform([x])[0] if x in self.label_encoder.classes_ else -1)
                df['ip_address'] = df['ip_address'].apply(
                    lambda x: self.label_encoder.transform([x])[0] if x in self.label_encoder.classes_ else -1)

        return df[['user', 'ip_address', 'success']]

    def analyze_log_lines(self, log_lines: List[str]) -> Dict[str, Any]:
        if not log_lines:
            return {"error": "No log lines provided"}

        try:
            parsed_entries = []
            for line in log_lines:
                parsed = self.parse_log_line(line)
                if parsed:
                    parsed_entries.append(parsed)

            if not parsed_entries:
                return {"error": "No valid log entries found"}

            features_df = self.extract_features(parsed_entries)

            if features_df.empty:
                return {"error": "No features could be extracted"}

            if self.feature_extractor:
                try:
                    features = self.feature_extractor.transform(features_df)
                except Exception as e:
                    logger.warning(f"Feature extraction failed, using raw features: {e}")
                    features = features_df.values
            else:
                features = features_df.values

            if self.anomaly_model:
                try:
                    scores = self.anomaly_model.predict_proba(features)[:, 1]
                    threshold = 0.5
                    predictions = (scores >= threshold).astype(int)

                    results = []
                    for i, entry in enumerate(parsed_entries):
                        results.append({
                            'line': entry['raw_line'],
                            'timestamp': entry.get('timestamp'),
                            'event_type': entry.get('event_type'),
                            'user': entry.get('user'),
                            'ip_address': entry.get('ip_address'),
                            'success': entry.get('success'),
                            'anomaly_score': float(scores[i]),
                            'is_anomaly': bool(predictions[i]),
                            'anomaly_severity': 'high' if scores[i] >= 0.9 else 'medium' if scores[i] >= 0.7 else 'low'
                        })

                    anomaly_count = sum(r['is_anomaly'] for r in results)
                    high_severity_count = sum(r['anomaly_severity'] == 'high' for r in results)

                    return {
                        'summary': {
                            'total_entries': len(results),
                            'anomalies_detected': anomaly_count,
                            'anomaly_percentage': (anomaly_count / len(results)) * 100,
                            'high_severity_anomalies': high_severity_count
                        },
                        'anomalies': [r for r in results if r['is_anomaly']],
                        'all_entries': results
                    }

                except Exception as e:
                    logger.error(f"Anomaly detection failed: {e}")
                    return {"error": f"Anomaly detection failed: {str(e)}"}

            else:
                return {"error": "Anomaly detection model not available"}

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
