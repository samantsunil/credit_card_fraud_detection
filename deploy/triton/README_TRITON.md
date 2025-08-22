# Triton deployment notes (deploy/triton)

This directory contains helper scripts and the Triton model repository used for local testing.

Files of interest
- cleanup_unwanted.sh — removes duplicate/legacy Triton client files.
- train_deploy_test.sh — trains the improved model and stages the artifact into deploy/triton/model_repository/<MODEL_NAME>/1/. This script no longer starts Triton automatically; it prints the PowerShell command to run Triton manually on Windows.
- model_repository/ — Triton model repository layout used for local testing.
- test_fraud_detection.py — comprehensive Triton test client (runnable directly with python; does not require pytest). The client converts numpy types to native Python values before JSON serialization.

Recommended workflow (Windows with PowerShell + Rancher Desktop)
1) Train and stage the model (from repo root)
```powershell
# run in Git Bash/WSL or via the provided script
DATA_PATH=data/creditcard.csv ./deploy/triton/train_deploy_test.sh
```
The script will:
- Train the improved Random Forest + SMOTE model using src/improved_retraining.py --deploy
- Attempt to copy the produced .joblib/.pkl artifact into deploy/triton/model_repository/<MODEL_NAME>/1/
- Print the PowerShell command you should run to start Triton and then wait for you to confirm Triton is running

2) Start Triton server (PowerShell example)
```powershell
$ModelRepo = Join-Path (Get-Location).Path "deploy\triton\model_repository"
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 `
  -v "$ModelRepo:/models" triton-scikit-learn:24.05-py3 `
  tritonserver --model-repository=/models --strict-model-config=false
```
- If you use Rancher Desktop/nerdctl and you built the image there, run the equivalent command inside the same WSL distro or replace `docker` with `nerdctl`.

3) Verify Triton readiness
```bash
curl -sf http://localhost:8000/v2/health/ready
```
Wait until the endpoint returns 200 before running tests.

4) Run the test client
```bash
python deploy/triton/test_fraud_detection.py
```
- The client sends several synthetic "valid" and "fraud" transactions to the model on Triton and prints JSON results.
- It converts numpy scalars/arrays to native Python types so JSON payloads work with Triton's HTTP API.

Model repository layout expected by Triton
- deploy/triton/model_repository/
  - fraud_rf_smote/
    - 1/
      - <model file>.joblib  (the training script places the artifact here)
    - config.pbtxt  (optional — Triton may auto-infer for some backends; include if required)

Notes and troubleshooting
- "image not found": ensure you run the container command in the same runtime/context that has the image (Windows Docker CLI vs Rancher Desktop containerd/nerdctl).
- If Triton exits immediately, re-run the container without `--rm` to inspect logs:
  - docker run --name triton_server ... (omit --rm) then docker logs -f triton_server
  - nerdctl run --name triton_server ... then nerdctl logs -f triton_server
- If the test client fails with JSON serialization errors, ensure the test script is up to date — it converts numpy types to native floats/ints automatically.

If you want the training script to also automatically start Triton in your environment, I can add an optional flag and implement runtime detection for your setup — let me know you prefer that.