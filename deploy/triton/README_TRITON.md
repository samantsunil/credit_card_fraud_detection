## Triton Inference Server Integration (Random Forest SMOTE)

This guide shows how to export the trained Random Forest (SMOTE) model and serve it with NVIDIA Triton Inference Server using the Python backend.

### 1) Train and export the model artifact

Run training; it produces `artifacts/models/random_forest_smote.joblib`.

```bash
python src/train_models.py --data-path data/creditcard.csv --test-size 0.3 --random-state 42
```

Copy the artifact into the Triton model version directory:

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
- inputs: `INPUT__0` float32 with dims `[30]` (feature count)
- outputs: `OUTPUT__PROB` (float32), `OUTPUT__CLASS` (int64)

### 3) Launch Triton

Option A: Docker (recommended)

```bash
docker run --rm --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/deploy/triton/model_repository:/models nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

CPU-only hosts can drop `--gpus=all`.

Option B: Native install — see Triton docs.

### 4) Send a test request

The input must be a 2D float32 array shaped `[batch, 30]`. Example with `curl` and JSON client:

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

### 5) Feature order and preprocessing

- The Kaggle dataset has 31 columns including label `Class`. The 30 features are: `Time`, `V1`..`V28`, `Amount`.
- This Triton model expects the same feature order used during training: the code trained with `X = df.drop(columns=["Class"]).values` (column order from the CSV).
- Ensure your client builds feature vectors in exactly this order. Alternatively, embed a column ordering step and scaler in the artifact.

### 6) Versioning and updates

- Upload new models as `deploy/triton/model_repository/fraud_rf_smote/<version>/` with updated `random_forest_smote.joblib`.
- Triton will hot-reload on repository changes if enabled.

### 7) References

- NVIDIA Triton Inference Server docs: `https://github.com/triton-inference-server/server`
- Python backend API: `https://github.com/triton-inference-server/python_backend`


