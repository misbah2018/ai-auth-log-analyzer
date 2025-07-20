import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create fake auth log features and labels for demo
data = {
    "ip": ["192.168.1.1", "192.168.1.2", "10.0.0.1"],
    "username": ["root", "admin", "guest"],
    "status": ["failed", "success", "failed"]
}
df = pd.DataFrame(data)

# Encode categorical variables
le_ip = LabelEncoder()
le_user = LabelEncoder()
le_status = LabelEncoder()

df["ip_enc"] = le_ip.fit_transform(df["ip"])
df["user_enc"] = le_user.fit_transform(df["username"])
df["status_enc"] = le_status.fit_transform(df["status"])

# Define features and targets
X = df[["ip_enc", "user_enc"]]
y_failed_login = (df["status"] == "failed").astype(int)         # Binary classification
y_intrusion = (df["username"] == "guest").astype(int)            # Dummy rule: guest = intrusion

# Train models
model_failed_login = RandomForestClassifier().fit(X, y_failed_login)
model_intrusion = RandomForestClassifier().fit(X, y_intrusion)

# Save models
os.makedirs("models", exist_ok=True)
with open("models/is_failed_login_model.pkl", "wb") as f:
    pickle.dump(model_failed_login, f)
with open("models/is_intrusion_model.pkl", "wb") as f:
    pickle.dump(model_intrusion, f)
with open("models/encoder.pkl", "wb") as f:
    pickle.dump({"ip": le_ip, "username": le_user}, f)

print("✅ Models trained and saved in /models")
