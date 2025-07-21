import pickle
from log_analyzer import AuthLogAnalyzer

# Load model and encoders
with open("../models/is_intrusion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

# Sample logs for testing
sample_logs = [
    {"user": "admin", "ip_address": "192.168.1.5", "success": 1},
    {"user": "user1", "ip_address": "192.168.1.10", "success": 1},
    {"user": "hacker", "ip_address": "203.0.113.50", "success": 0},   # anomalous
    {"user": "unknown", "ip_address": "10.0.0.8", "success": 0},     # anomalous
    {"user": "user2", "ip_address": "192.168.1.15", "success": 1}
]

# Create analyzer
analyzer = AuthLogAnalyzer(anomaly_model=model, label_encoders=encoders)

# Analyze logs
result = analyzer.detect_intrusions(sample_logs)

# Print summary and results
if "summary" in result:
    print("=== Summary ===")
    print(result["summary"])
    print("\n=== Detailed Results ===")
    for r in result["results"]:
        print(f"{r['entry']} => Anomaly: {r['is_anomaly']}, Score: {r['anomaly_score']:.4f}")
else:
    print("❌ Error:", result["error"])
