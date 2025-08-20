## Triton Inference Server Integration (Random Forest SMOTE)

This guide shows how to export the trained Random Forest (SMOTE) model and serve it with NVIDIA Triton Inference Server using the Python backend.

### 1) Train and export the model artifact

**Option A: Basic Training**
Run basic training; it produces `artifacts/models/random_forest_smote.joblib`.

```bash
python src/train_models.py --data-path data/creditcard.csv --test-size 0.3 --random-state 42
```

**Option B: Advanced Training (Recommended)**
Run improved training with feature engineering and advanced resampling:

```bash
python src/improved_retraining.py --data-path data/creditcard.csv --deploy
```

This automatically copies the improved model to the Triton directory.

**Manual copy (if needed):**
```bash
mkdir -p deploy/triton/model_repository/fraud_rf_smote/1
cp artifacts/models/random_forest_smote.joblib deploy/triton/model_repository/fraud_rf_smote/1/
```

### 2) Model repository layout

```
deploy/
  triton/
    model_repository/
      fraud_rf_smote/
        config.pbtxt
        1/
          model.py
          random_forest_smote.joblib
```

`config.pbtxt` defines:

- backend: `python`
- inputs: `INPUT__0` float32 with dims `[35]` (30 original features + 5 engineered features)
- outputs: `OUTPUT__PROB` (float32), `OUTPUT__CLASS` (int64)

**Note:** The improved model uses 35 features instead of 30. If using the basic model, change `dims: [35]` to `dims: [30]` in `config.pbtxt`.

### 3) Launch Triton

Option A: Docker (recommended)

**Step 1: Build custom Triton image with dependencies**
```bash
# Build the custom image with scikit-learn dependencies
docker build -t triton-scikit-learn:24.05-py3 deploy/triton/
```

**Step 2: Run Triton with custom image**

**Windows with Rancher Desktop (WSL):**
```bash
# Option 1: Using relative path (recommended)
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "$(pwd)/deploy/triton/model_repository:/models" triton-scikit-learn:24.05-py3 \
  tritonserver --model-repository=/models

# Option 2: Using relative path with dot notation
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "./deploy/triton/model_repository:/models" triton-scikit-learn:24.05-py3 \
  tritonserver --model-repository=/models

# Option 3: Copy model to a simple path and mount (workaround for path issues)
mkdir -p /tmp/triton_models
cp -r deploy/triton/model_repository/* /tmp/triton_models/
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "/tmp/triton_models:/models" triton-scikit-learn:24.05-py3 \
  tritonserver --model-repository=/models

# Option 4: Use PowerShell instead of Git Bash
# Open PowerShell and run:
# docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 `
#   -v "$(pwd)/deploy/triton/model_repository:/models" triton-scikit-learn:24.05-py3 `
#   tritonserver --model-repository=/models
```

**Linux/macOS:**
```bash
docker run --rm --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/deploy/triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

**Note:** 
- Rancher Desktop users: Try Option 1 first, then Option 2 or 3 if path issues occur
- Windows users: Omit `--gpus=all` (CPU-only)
- Linux/macOS with GPU: Include `--gpus=all`
- Linux/macOS CPU-only: Omit `--gpus=all`

Option B: Native install — see Triton docs.

### 4) Test the deployed model

**Option A: Using Python test scripts (Recommended)**

1. **Basic test with random data:**
```bash
python deploy/triton/test_client.py
```

2. **Comprehensive fraud detection test:**
```bash
python deploy/triton/test_fraud_detection.py
```

This script tests:
- 5 valid transactions (should be predicted as legitimate)
- 5 fraud transactions (should be predicted as fraud)
- Edge cases (zero values, large values, negative values)
- Provides detailed accuracy metrics

**Option B: Using curl with JSON**

The input must be a 2D float32 array shaped `[batch, 35]` (for improved model) or `[batch, 30]` (for basic model).

**Windows (Git Bash) - Basic Model (30 features):**
```bash
cat > sample.json <<'JSON'
{
  "inputs": [
    {
      "name": "INPUT__0",
      "datatype": "FP32",
      "shape": [1, 30],
      "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
  ],
  "outputs": [
    {"name": "OUTPUT__PROB"},
    {"name": "OUTPUT__CLASS"}
  ]
}
JSON

curl -s -X POST http://localhost:8000/v2/models/fraud_rf_smote/infer \
  -H 'Content-Type: application/json' \
  -d @sample.json
```

**Windows (Git Bash) - Improved Model (35 features):**
```bash
cat > sample_improved.json <<'JSON'
{
  "inputs": [
    {
      "name": "INPUT__0",
      "datatype": "FP32",
      "shape": [1, 35],
      "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
  ],
  "outputs": [
    {"name": "OUTPUT__PROB"},
    {"name": "OUTPUT__CLASS"}
  ]
}
JSON

curl -s -X POST http://localhost:8000/v2/models/fraud_rf_smote/infer \
  -H 'Content-Type: application/json' \
  -d @sample_improved.json
```

**Linux/macOS (with jq) - Basic Model (30 features):**
```bash
cat > sample.json <<'JSON'
{
  "inputs": [
    {
      "name": "INPUT__0",
      "datatype": "FP32",
      "shape": [1, 30],
      "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
  ],
  "outputs": [
    {"name": "OUTPUT__PROB"},
    {"name": "OUTPUT__CLASS"}
  ]
}
JSON

curl -s -X POST http://localhost:8000/v2/models/fraud_rf_smote/infer \
  -H 'Content-Type: application/json' \
  -d @sample.json | jq .
```

**Linux/macOS (with jq) - Improved Model (35 features):**
```bash
cat > sample_improved.json <<'JSON'
{
  "inputs": [
    {
      "name": "INPUT__0",
      "datatype": "FP32",
      "shape": [1, 35],
      "data": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    }
  ],
  "outputs": [
    {"name": "OUTPUT__PROB"},
    {"name": "OUTPUT__CLASS"}
  ]
}
JSON

curl -s -X POST http://localhost:8000/v2/models/fraud_rf_smote/infer \
  -H 'Content-Type: application/json' \
  -d @sample_improved.json | jq .
```

### 5) Feature order and preprocessing

**Basic Model (30 features):**
- The Kaggle dataset has 31 columns including label `Class`. The 30 features are: `Time`, `V1`..`V28`, `Amount`.
- This Triton model expects the same feature order used during training: the code trained with `X = df.drop(columns=["Class"]).values` (column order from the CSV).

**Improved Model (35 features):**
- Original 30 features: `Time`, `V1`..`V28`, `Amount`
- Plus 5 engineered features:
  1. Sum of absolute V-values (overall magnitude)
  2. Standard deviation of V-values (variability)
  3. Maximum absolute V-value (extreme values)
  4. Amount-to-time ratio (unusual timing patterns)
  5. Number of outliers (V-features with abs value > 2)

- Ensure your client builds feature vectors in exactly this order. The Python test scripts automatically calculate the engineered features.

### 6) Versioning and updates

- Upload new models as `deploy/triton/model_repository/fraud_rf_smote/<version>/` with updated `random_forest_smote.joblib`.
- Triton will hot-reload on repository changes if enabled.
- When switching between basic (30 features) and improved (35 features) models, remember to update the `dims` in `config.pbtxt`.

### 7) Testing Summary

**Quick Testing Workflow:**
1. **Train and deploy:** `python src/improved_retraining.py --data-path data/creditcard.csv --deploy`
2. **Start Triton:** Use one of the Docker commands from section 3
3. **Test the model:** `python deploy/triton/test_fraud_detection.py`

**Expected Results:**
- Valid transactions should be predicted as "LEGITIMATE" (class 0)
- Fraud transactions should be predicted as "FRAUD" (class 1)
- The comprehensive test script provides accuracy metrics and detailed analysis

### 7) References

- NVIDIA Triton Inference Server docs: `https://github.com/triton-inference-server/server`
- Python backend API: `https://github.com/triton-inference-server/python_backend`

### 8) Troubleshooting

**Path issues with Rancher Desktop (WSL):**
- **Option 1**: Try the relative path first: `"$(pwd)/deploy/triton/model_repository:/models"`
- **Option 2**: If that fails, use dot notation: `"./deploy/triton/model_repository:/models"`
- **Option 3**: Use the `/tmp` copy workaround (Option 3 in the Docker commands above)
- Check that `random_forest_smote.joblib` exists in `deploy/triton/model_repository/fraud_rf_smote/1/`

**Verify model repository structure:**
```bash
ls -la deploy/triton/model_repository/fraud_rf_smote/
ls -la deploy/triton/model_repository/fraud_rf_smote/1/
```

**Rancher Desktop specific checks:**
```bash
# Check if Docker is running
docker ps

# Test volume mounting with a simple container (try these in order)
docker run --rm -v "$(pwd)/deploy/triton/model_repository:/test" alpine ls -la /test

# Test with dot notation if $(pwd) fails
docker run --rm -v "./deploy/triton/model_repository:/test" alpine ls -la /test

# Check WSL integration
wsl --list --verbose

# Verify the model repository exists and has correct structure
ls -la deploy/triton/model_repository/fraud_rf_smote/
ls -la deploy/triton/model_repository/fraud_rf_smote/1/
```

- NVIDIA Triton Inference Server docs: `https://github.com/triton-inference-server/server`
- Python backend API: `https://github.com/triton-inference-server/python_backend`


