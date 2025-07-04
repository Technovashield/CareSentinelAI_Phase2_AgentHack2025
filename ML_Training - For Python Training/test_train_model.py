from train_model import train_model
import os

# Define paths
latest_input_csv = r"C:\Users\PatelR\OneDrive\Documents\UiPath\CareSentinelAI_Phase2\OutputData\MergedFinalOutput\MasterPatientMetrics_20250611_201253.csv"

output_model_path = r"C:\Users\PatelR\OneDrive\Documents\UiPath\CareSentinelAI_Phase2\ML_Training - For Python Training\OutputTrainedPredict\RiskClassifier.pkl"

# Train the model
train_model(
    input_csv_path=latest_input_csv,
    output_model_path=output_model_path
)

# Confirm it was saved
if os.path.exists(output_model_path):
    print(f"[OK] Model saved successfully at {output_model_path}")
else:
    print(f"[ERROR] Model NOT created.")
