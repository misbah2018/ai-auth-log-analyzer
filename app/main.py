# app/main.py

import argparse
import json
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(__file__))

from model import load_model, predict_anomalies
from utils import parse_log_file, preprocess_df


def main():
    parser = argparse.ArgumentParser(description="Auth Log Anomaly Analyzer")
    parser.add_argument("--input", type=str, required=True, help="Path to input log or CSV file")
    parser.add_argument("--output", type=str, help="Path to save output JSON")
    args = parser.parse_args()

    # Step 1: Load input
    if args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        df = parse_log_file(args.input)

    if df.empty:
        print("❌ No valid log entries found.")
        return

    # Step 2: Preprocess features
    X, original_df = preprocess_df(df)

    # Step 3: Load model and predict
    model = load_model()
    results = predict_anomalies(model, X, original_df)

    # Step 4: Show summary
    print("\n=== Anomaly Summary ===")
    print(json.dumps(results["summary"], indent=2))
    
    print("\n=== Sample Anomalies ===")
    for anomaly in results["anomalies"][:5]:
        print(anomaly)

    # Step 5: Optional output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
