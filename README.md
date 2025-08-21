# Credit Card Fraud Detection (Python)

This project trains a Random Forest classifier for credit card fraud detection using SMOTE to handle class imbalance. It includes training, evaluation, model artifact export and optional deployment to NVIDIA Triton Inference Server.

Quick: train, stage model, start Triton and test
- Use the helper scripts in deploy/triton/ to automate training and staging the model into a Triton model repository.
- Starting Triton is done manually (PowerShell on Windows) because Rancher Desktop / local environments differ; the script prints the exact command to run.

1) Make scripts executable (from Git Bash / WSL)
```bash
chmod +x deploy/triton/*.sh
```

2) Optional: remove duplicate Triton clients
```bash
./deploy/triton/cleanup_unwanted.sh
```

3) Train and stage model (this script trains and copies the artifact into deploy/triton/model_repository/<MODEL_NAME>/1/)
```bash
# from repo root; override DATA_PATH if needed
DATA_PATH=data/creditcard.csv ./deploy/triton/train_deploy_test.sh
```

4) Start Triton server (manual step — PowerShell example)
- On Windows / PowerShell (run from the repo root; this maps the host model repo into the container):
```powershell
# PowerShell example — adjust image name if different
$ModelRepo = Join-Path (Get-Location).Path "deploy\triton\model_repository"
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 `
  -v "$ModelRepo:/models" triton-scikit-learn:24.05-py3 `
  tritonserver --model-repository=/models --strict-model-config=false
```

- On WSL / Linux (or if using nerdctl with Rancher Desktop inside WSL):
```bash
# docker (or nerdctl) example
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "$(pwd)/deploy/triton/model_repository:/models" triton-scikit-learn:24.05-py3 \
  tritonserver --model-repository=/models --strict-model-config=false
```

Note: If you built the Triton image inside Rancher Desktop, run the container command in the same context (WSL distro) so the runtime can see the image. If you use Rancher Desktop's containerd/nerdctl, replace `docker` with `nerdctl` in the command.

5) Test the deployed model (once Triton is ready)
```bash
python deploy/triton/test_fraud_detection.py
```
- The test script is runnable directly (does not require pytest installed), converts numpy types to native Python types for JSON, and will attempt HTTP inference against http://localhost:8000 by default.
- If you prefer pytest-style execution, install pytest and run it under pytest; the script is written to be usable both ways.

Environment variables / overrides
- DATA_PATH — path to creditcard.csv (default: data/creditcard.csv)
- MODEL_REPO — path to model repository used by scripts (default: $(pwd)/deploy/triton/model_repository)
- MODEL_NAME — model name inside model repository (default: fraud_rf_smote)
- TRITON_IMAGE — override Triton image name if different from triton-scikit-learn:24.05-py3

Troubleshooting
- If the script cannot find your Triton image, list local images:
  - docker: docker images | grep -i triton
  - nerdctl: nerdctl images | grep -i triton
- Ensure the model artifact exists under deploy/triton/model_repository/<MODEL_NAME>/1/ (train script tries to stage it).
- If the Triton container exits early, inspect logs:
  - docker: docker logs <container_name>
  - nerdctl: nerdctl logs <container_name>

Manual flow summary
- Train & stage: DATA_PATH=... ./deploy/triton/train_deploy_test.sh
- Start Triton (PowerShell or WSL as shown)
- Run test client: python deploy/triton/test_fraud_detection.py

