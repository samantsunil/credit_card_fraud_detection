#!/usr/bin/env bash
set -euo pipefail

# ...existing code...

DATA_PATH="${DATA_PATH:-data/creditcard.csv}"
MODEL_REPO="${MODEL_REPO:-$(pwd)/deploy/triton/model_repository}"
MODEL_NAME="${MODEL_NAME:-fraud_rf_smote}"

echo "DATA_PATH=$DATA_PATH"
echo "MODEL_REPO=$MODEL_REPO"
echo "MODEL_NAME=$MODEL_NAME"

if [ ! -f "$DATA_PATH" ]; then
  echo "Error: dataset not found at $DATA_PATH"
  exit 1
fi

echo "1) Train improved model (this should save artifacts and optionally copy into MODEL_REPO when --deploy is supported)"
python src/improved_retraining.py --data-path "$DATA_PATH" --deploy

sleep 1

echo "Locating produced model artifact..."
CANDIDATES=()
while IFS= read -r -d $'\0' f; do CANDIDATES+=("$f"); done < <(find artifacts -type f \( -iname "*.joblib" -o -iname "*.pkl" \) -print0 2>/dev/null || true)
while IFS= read -r -d $'\0' f; do CANDIDATES+=("$f"); done < <(find models -type f \( -iname "*.joblib" -o -iname "*.pkl" \) -print0 2>/dev/null || true)

if [ "${#CANDIDATES[@]}" -eq 0 ]; then
  echo "No model artifacts found in artifacts/ or models/."
  echo "If improved_retraining.py --deploy should have copied the model into the Triton repo automatically, check that step."
else
  echo "Found candidate model files:"
  for c in "${CANDIDATES[@]}"; do echo " - $c"; done
fi

TARGET_DIR="$MODEL_REPO/$MODEL_NAME/1"
mkdir -p "$TARGET_DIR"

MODEL_IN_REPO="$(find "$MODEL_REPO" -type f \( -iname "*.joblib" -o -iname "*.pkl" \) -path "*/$MODEL_NAME/1/*" -print -quit 2>/dev/null || true)"

if [ -n "$MODEL_IN_REPO" ]; then
  echo "Model already present in model repository: $MODEL_IN_REPO"
else
  if [ "${#CANDIDATES[@]}" -gt 0 ]; then
    SRC="${CANDIDATES[0]}"
    BASENAME="$(basename "$SRC")"
    DEST="$TARGET_DIR/$BASENAME"
    echo "Copying model $SRC -> $DEST"
    cp -v "$SRC" "$DEST"
    MODEL_IN_REPO="$DEST"
  else
    echo "No model to copy into model repository. Aborting."
    exit 1
  fi
fi

echo "Model staged at: $MODEL_IN_REPO"

# Removed automatic Triton start/stop. User will run Triton separately (PowerShell example below).
echo
echo "===================================================="
echo "Start Triton server manually from PowerShell (example):"
echo
echo 'docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 `'
echo '  -v "$(Get-Location).Path\deploy\triton\model_repository:/models" triton-scikit-learn:24.05-py3 `'
echo '  tritonserver --model-repository=/models'
echo
echo "Note: run the above command from a PowerShell session on Windows."
echo "Once Triton is started, return to this script and press ENTER to run the test client."
echo "===================================================="
echo
read -r -p "Press ENTER once Triton is running and reachable on http://localhost:8000 ..."

echo "Checking Triton readiness (waiting up to 60s)..."
for i in $(seq 1 60); do
  if curl -sfS http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
    echo "Triton is ready."
    break
  fi
  sleep 1
done

if ! curl -sfS http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
  echo "Triton did not become ready in time. Please check the Triton server logs on your Windows host."
  exit 1
fi

echo "Running Triton test client: deploy/triton/test_fraud_detection.py"
python deploy/triton/test_fraud_detection.py || {
  echo "Triton test client returned non-zero exit code. Check output above."
  exit 1
}

echo "Completed tests. (This script does not stop the Triton server — stop it from PowerShell when done.)"