import os
import pandas as pd
import joblib
from datetime import datetime
import sklearn
import sys

# === Logger ===
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {msg}")
    with open("C:\\Users\\PatelR\\OneDrive\\Documents\\UiPath\\CareSentinelAI_Phase2\\OutputData\\Predictions\\script_trace.txt", "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {msg}\n")

# === Version Info ===
log(f"[DEBUG] Working directory: {os.getcwd()}")
log(f"[DEBUG] Script path: {os.path.abspath(__file__)}")
log(f"[TRACE] Python executable: {sys.executable}")
log(f"[TRACE] scikit-learn version: {sklearn.__version__}")
log(f"[TRACE] sklearn location: {sklearn.__file__}")
log(f"[TRACE] sys.path: {sys.path}")
log("[INFO] Script started")

# === Get latest file by prefix ===
def get_latest_file(folder, prefix, extension):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(extension)]
    if not files:
        raise FileNotFoundError(f"No matching files with prefix '{prefix}' in {folder}")
    return max(files, key=os.path.getmtime)

# === Define Paths ===
base_path = r"C:\Users\PatelR\OneDrive\Documents\UiPath\CareSentinelAI_Phase2"
metrics_folder = os.path.join(base_path, "OutputData", "MergedFinalOutput")
model_folder = os.path.join(base_path, "ML_Training - For Python Training", "OutputTrainedPredict")
predictions_folder = os.path.join(base_path, "OutputData", "Predictions")

# === Detect Input and Model Files ===
try:
    input_csv = get_latest_file(metrics_folder, "MasterPatientMetrics_", ".csv")
    model_path = get_latest_file(model_folder, "RiskClassifier", ".pkl")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(predictions_folder, f"PatientRiskPredictions_{timestamp}.csv")
except Exception as e:
    log(f"[ERROR] File detection failed: {e}")
    sys.exit(1)

log(f"[INPUT] CSV path: {input_csv}")
log(f"[MODEL] Model path: {model_path}")
log(f"[OUTPUT] Prediction output path: {output_csv}")

# === Load CSV ===
try:
    df = pd.read_csv(input_csv)
    log("[OK] Input CSV loaded")
except Exception as e:
    log(f"[ERROR] Failed to load input CSV: {e}")
    sys.exit(1)

# === Compute Age from DOB ===
try:
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    df["Age"] = df["DOB"].apply(lambda dob: (datetime.now() - dob).days // 365 if pd.notnull(dob) else None)
    log("[OK] Age computed from DOB")
except Exception as e:
    log(f"[ERROR] Failed to compute Age: {e}")
    sys.exit(1)

# === Load Model ===
try:
    model = joblib.load(model_path)
    log(f"[OK] Model loaded: {type(model)}")
except Exception as e:
    log(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# === Features & Prediction ===
features = ["BMI", "Steps", "SleepHours", "HeartRate", "Age", "Cholesterol"]
missing = [col for col in features if col not in df.columns]

if missing:
    log(f"[ERROR] Missing required features: {missing}")
    sys.exit(1)

df_clean = df.dropna(subset=features)

if df_clean.empty:
    log("[ERROR] No valid rows after dropping NaNs.")
    sys.exit(1)

try:
    df_clean["PredictedRisk"] = model.predict(df_clean[features])
    df_clean.to_csv(output_csv, index=False)
    log("[SUCCESS] Prediction completed and saved.")
except Exception as e:
    log(f"[ERROR] Prediction failed: {e}")
    sys.exit(1)


