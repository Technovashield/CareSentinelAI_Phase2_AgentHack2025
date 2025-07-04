# CareSentinelAI - Predictive Health Risk Mitigation (Phase 2)

**Submitted for:** UiPath AgentHack 2025  
**Developed by:** Roopa Patel, RPA Developer & Product Owner at Technovashield Limited

---

## 📌 Overview

CareSentinelAI is an AI-powered health monitoring solution designed to detect early warning signs of health deterioration by integrating wearable data, lab results, and patient records.

This **Phase 2** implementation focuses on building a fully automated health risk prediction pipeline using **UiPath Agentic Automation and Maestro** to simulate real-world clinical decisions.

---

## ⚙️ Technologies Used

- UiPath Studio & Orchestrator
- UiPath AI Center & Labs (Maestro)
- Python (scikit-learn, pandas)
- CSV/Excel for reporting

---

# 📁 Folder Structure
CareSentinelAI_Phase2_AgentHack2025/
├── Workflows/                   # UiPath workflows for metrics, ML, alerting
├── PythonModel/                 # ML training and prediction scripts
├── MergedFinalOutput/          # Generated predictions and Excel dashboards
├── SubmissionAssets/           # Docs: Presentation, Checklist, Enhancement Plan
├── README.md
├── .gitignore
└── LICENSE
---

## ✅ Key Features

- Unified patient identity and metrics from multiple sources
- ML-based risk scoring using `HistGradientBoostingClassifier`
- Automated email alerts for high-risk patients
- Timestamped output reports and charts
- UiPath Agent simulates real-time decision-making

---

## ▶️ How to Run the Demo

1. Open `Main.xaml` in UiPath Studio.
2. Ensure all file paths in config or arguments are valid.
3. Run the full process (data → ML prediction → reporting → email).
4. To retrain the model, run `train_model.py` and then `predict_risk.py`.

---

## 📜 License

This project is licensed under the **MIT License**.  
© 2025 Technovashield Limited

---

## 🔗 Submission Link

[Forum Post (to be added)](#)


