import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
import sys
import os
from datetime import datetime
import sklearn

# === Utility Logger ===
def log(msg):
    with open("script_trace_train.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - " + str(msg) + "\n")

# === Version Check ===
log(f"[TRACE] Python executable: {sys.executable}")
log(f"[TRACE] scikit-learn version: {sklearn.__version__}")
log(f"[TRACE] sklearn location: {sklearn.__file__}")
log(f"[TRACE] sys.path: {sys.path}")

# === Entry Point ===
def train_model(input_csv_path, output_model_path):
    log("[INFO] Training started")
    log(f"[INPUT] CSV path: {input_csv_path}")
    log(f"[OUTPUT] Model path: {output_model_path}")

    df = pd.read_csv(input_csv_path)

    if "DOB" in df.columns and "Age" not in df.columns:
        log("[INFO] Computing Age from DOB")
        df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
        df["Age"] = df["DOB"].apply(lambda dob: (pd.Timestamp.now() - dob).days // 365 if pd.notnull(dob) else None)

    features = ["BMI", "Steps", "SleepHours", "HeartRate", "Age", "Cholesterol"]

    if not all(col in df.columns for col in features + ["RiskLevel"]):
        log("[ERROR] Missing one or more required columns in input CSV")
        missing_cols = [col for col in features + ["RiskLevel"] if col not in df.columns]
        log(f"[ERROR] Missing columns: {missing_cols}")
        return
    

    df = df.dropna(subset=features + ["RiskLevel"])
    X = df[features]
    y = df["RiskLevel"]

    clf = HistGradientBoostingClassifier()
    clf.fit(X, y)

    joblib.dump(clf, output_model_path)
    log("[SUCCESS] Model trained and saved successfully")

# === Script Trigger ===
if __name__ == "__main__":
    if len(sys.argv) != 3:
        log("[ERROR] Invalid number of arguments. Expected: <input_csv> <output_model>")
    else:
        input_csv = sys.argv[1]
        output_model = sys.argv[2]
        train_model(input_csv, output_model)
