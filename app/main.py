# app/main.py

import argparse
import pandas as pd
import json
from app.model import load_model, predict_anomalies
from app.utils import parse_log_file, preprocess_df


def run_analysis(input_path: str, output_path: str = "temp_result.json") -> dict:
    print("✅ Starting analysis...")

    # Step 1: Load input
    try:
        if input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        else:
            df = parse_log_file(input_path)
    except Exception as e:
        print(f"❌ Error loading file {input_path}: {e}")
        raise  # ✅ Re-raises the actual exception correctly

    if df.empty:
        raise ValueError("❌ No valid log entries found.")

    print("📊 Log data loaded successfully.")

    # Step 2: Preprocess features
    X, original_df = preprocess_df(df)
    print("🔧 Data preprocessing completed.")

    # Step 3: Load model and predict
    model = load_model()
    results = predict_anomalies(model, X, original_df)
    print("🤖 Anomaly detection complete.")

    # Step 4: Save output if requested
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {output_path}")
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Auth Log Anomaly Analyzer")
    parser.add_argument("--input", type=str, required=True, help="Path to input log or CSV file")
    parser.add_argument("--output", type=str, help="Path to save output JSON")
    args = parser.parse_args()

    try:
        results = run_analysis(args.input, args.output)
        print("\n=== 📈 Anomaly Summary ===")
        print(json.dumps(results.get("summary", {}), indent=2))

        print("\n=== 🚨 Sample Anomalies ===")
        for anomaly in results.get("anomalies", [])[:5]:
            print(anomaly)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
