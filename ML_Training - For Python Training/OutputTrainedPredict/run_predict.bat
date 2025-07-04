@echo off
echo [TRACE] Running from folder: %CD% >> "C:\Users\PatelR\OneDrive\Documents\UiPath\CareSentinelAI_Phase2\OutputData\Predictions\script_trace.txt"

REM Log start
echo [INFO] Batch file triggered at %date% %time% >> "C:\Users\PatelR\OneDrive\Documents\UiPath\CareSentinelAI_Phase2\OutputData\Predictions\script_trace.txt"

python "C:\Users\PatelR\OneDrive\Documents\UiPath\CareSentinelAI_Phase2\ML_Training - For Python Training\OutputTrainedPredict\predict_risk.py"

REM Log complete
echo [INFO] Script complete at %date% %time% >> "C:\Users\PatelR\OneDrive\Documents\UiPath\CareSentinelAI_Phase2\OutputData\Predictions\script_trace.txt"
pause
