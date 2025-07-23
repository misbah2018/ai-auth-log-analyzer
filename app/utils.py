import pandas as pd

def parse_log_file(filepath):
    entries = []
    with open(filepath, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                ip = parts[0]
                user = parts[1]
                success = 1 if parts[2].lower() == "success" else 0
                entries.append({"ip_address": ip, "user": user, "success": success})
    return pd.DataFrame(entries)

def preprocess_df(df):
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

    expected_cols = ["user_success_rate", "ip_login_count", "is_internal_ip", "user_login_count"]
    return df[expected_cols], df
