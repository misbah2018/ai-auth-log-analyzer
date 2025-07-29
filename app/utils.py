# app/utils.py

import re
import pandas as pd


def parse_log_file(filepath):
    """
    Parses SSH log lines for login attempts (Accepted/Failed) and extracts key details.
    """
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            match = re.search(
                r'(?P<month>\w{3}) (?P<day>\d{1,2}) .*sshd.* (?P<status>Accepted|Failed) password for (?P<user>\w+) from (?P<ip>\d+\.\d+\.\d+\.\d+)'
,
                line
            )
            if match:
                data = match.groupdict()
                timestamp = f"{data['month']} {data['day']}"
                entries.append({
                    "timestamp": timestamp,
                    "event_type": "login_attempt",
                    "user": data["user"],
                    "ip_address": data["ip"],
                    "success": 1 if data["status"] == "Accepted" else 0,
                })

    return pd.DataFrame(entries)


def preprocess_df(df):
    """
    Transforms raw log DataFrame into a feature set for model prediction.
    """
    # Ensure correct types
    df["ip_address"] = df["ip_address"].astype(str)
    df["user"] = df["user"].astype(str)
    df["success"] = df["success"].astype(int)

    # Feature engineering
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

    # Select model input features
    expected_cols = ["user_success_rate", "ip_login_count", "is_internal_ip", "user_login_count"]
    return df[expected_cols], df
