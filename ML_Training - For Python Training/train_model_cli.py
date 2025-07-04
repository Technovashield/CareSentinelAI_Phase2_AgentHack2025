# train_model_cli.py
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# === 1. Load Data ===
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# === 2. Train Model ===
clf = LogisticRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

# === 3. Define Output Path ===
output_dir = "ML_Training - For Python Training/Models/OutputModels"
os.makedirs(output_dir, exist_ok=True)

# === 4. Write Model Result ===
output = {"accuracy": accuracy}
with open(os.path.join(output_dir, "model_result.json"), "w") as f:
    json.dump(output, f)

# === 5. (Optional) Write Execution Log ===
with open(os.path.join(output_dir, "log.txt"), "w") as log:
    log.write("Script executed successfully.\n")
    log.write(f"Accuracy: {accuracy}")
