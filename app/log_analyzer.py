import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Simulate parsed log entries
parsed_entries = [
    {"timestamp": "Jul 21 10:12:34", "event_type": "login_attempt", "user": "user1", "ip_address": "192.168.1.10", "success": 1},
    {"timestamp": "Jul 21 10:12:50", "event_type": "login_attempt", "user": "admin", "ip_address": "10.0.0.5", "success": 1},
    {"timestamp": "Jul 21 10:13:10", "event_type": "login_attempt", "user": "user2", "ip_address": "192.168.1.15", "success": 0},
    {"timestamp": "Jul 21 10:13:25", "event_type": "login_attempt", "user": "sysadmin", "ip_address": "192.168.1.7", "success": 1},
    {"timestamp": "Jul 21 10:14:01", "event_type": "login_attempt", "user": "root", "ip_address": "45.33.32.156", "success": 0},
    {"timestamp": "Jul 21 10:14:10", "event_type": "login_attempt", "user": "unknown", "ip_address": "203.0.113.5", "success": 0},
    {"timestamp": "Jul 21 10:14:11", "event_type": "login_attempt", "user": "unknown", "ip_address": "203.0.113.5", "success": 0},
    {"timestamp": "Jul 21 10:14:12", "event_type": "login_attempt", "user": "unknown", "ip_address": "203.0.113.5", "success": 0},
    {"timestamp": "Jul 21 10:14:13", "event_type": "login_attempt", "user": "unknown", "ip_address": "203.0.113.5", "success": 0},
    {"timestamp": "Jul 21 10:15:00", "event_type": "login_attempt", "user": "user1", "ip_address": "192.168.1.10", "success": 1},
]

# Extract all users and IPs for feature mapping
users = [e["user"] for e in parsed_entries]
ips = [e["ip_address"] for e in parsed_entries]

user_counts = pd.Series(users).value_counts().to_dict()
ip_counts = pd.Series(ips).value_counts().to_dict()
success_map = pd.DataFrame(parsed_entries).groupby("user")["success"].mean().to_dict()

def is_internal(ip):
    return ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172.")

# --- Build feature DataFrame ---
for entry in parsed_entries:
    entry["ip_login_count"] = ip_counts.get(entry["ip_address"], 0)
    entry["user_login_count"] = user_counts.get(entry["user"], 0)
    entry["user_success_rate"] = success_map.get(entry["user"], 0.0)
    entry["is_internal_ip"] = int(is_internal(entry["ip_address"]))

features_df = pd.DataFrame(parsed_entries)
expected_cols = ["user_success_rate", "ip_login_count", "is_internal_ip", "user_login_count"]

# --- Load the model ---
with open("models/is_intrusion_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Predict ---
scores = model.decision_function(features_df[expected_cols])
predictions = model.predict(features_df[expected_cols])  # -1 = anomaly, 1 = normal

# --- Compute thresholds for severity ---
threshold_high = np.percentile(scores, 10)
threshold_medium = np.percentile(scores, 30)

# --- Tag anomalies ---
anomalies = []
all_tagged = []
for entry, score, pred in zip(parsed_entries, scores, predictions):
    is_anomaly = pred == -1

    # Dynamic severity logic
    if score < threshold_high:
        severity = "high"
    elif score < threshold_medium:
        severity = "medium"
    else:
        severity = "low"

    entry.update({
        "anomaly_score": score,
        "is_anomaly": is_anomaly,
        "anomaly_severity": severity if is_anomaly else "none"
    })

    all_tagged.append(entry)
    if is_anomaly:
        anomalies.append(entry)

# --- Summary Report ---
summary = {
    "total_entries": len(parsed_entries),
    "anomalies_detected": len(anomalies),
    "anomaly_percentage": round(len(anomalies) / len(parsed_entries) * 100, 2),
    "high_severity_anomalies": sum(1 for a in anomalies if a["anomaly_severity"] == "high"),
}

output = {
    "summary": summary,
    "anomalies": anomalies,
    "all_entries": all_tagged
}

import json
print("📊 Final anomaly report:")
print(json.dumps(output, indent=2))
