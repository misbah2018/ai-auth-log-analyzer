import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os
import random

# Simulate normal user behavior
normal_users = ["admin", "user1", "user2", "ubuntu", "sysadmin"]
normal_ips = ["192.168.1." + str(i) for i in range(1, 21)] + ["10.0.0." + str(i) for i in range(1, 11)]

# Generate 500 logins (mostly successful)
data = []
for _ in range(500):
    data.append({
        "ip_address": random.choice(normal_ips),
        "user": random.choice(normal_users),
        "success": np.random.choice([1, 0], p=[0.9, 0.1])  # mostly successful
    })

df = pd.DataFrame(data)

# --- Feature Engineering ---
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

# ✅ Consistent feature order
expected_cols = ["user_success_rate", "ip_login_count", "is_internal_ip", "user_login_count"]

# Train on mostly successful + a few failures
success_df = df[df["success"] == 1]
few_failures = df[df["success"] == 0].sample(frac=0.1, random_state=42)
training_data = pd.concat([success_df, few_failures])
X_train = training_data[expected_cols]

# Print training feature sample
print("🔧 Training feature sample:")
print(X_train.head())

# --- Train the model ---
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# --- Save the model ---
os.makedirs("models", exist_ok=True)
with open("models/is_intrusion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved.")
