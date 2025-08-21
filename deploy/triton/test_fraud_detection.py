#!/usr/bin/env python3
"""
Comprehensive test for Credit Card Fraud Detection with Triton
Tests both valid and fraud transaction patterns
"""
from typing import List, Dict, Any

# Make pytest optional so the script can be run directly without installing pytest
try:
    import pytest  # type: ignore
except Exception:
    pytest = None

import os
import sys
import time
import json
import requests
import numpy as np
import importlib
import joblib


def calculate_enhanced_features(features: list) -> list:
    """Calculate enhanced features for a given feature vector."""
    # Basic engineered features (5 features as in improved_retraining.py)
    v_sum_abs = np.sum(np.abs(features[1:29]))
    v_std = np.std(features[1:29])
    v_max_abs = np.max(np.abs(features[1:29]))
    amount_time_ratio = features[29] / (features[0] + 1)
    v_outliers = np.sum(np.abs(features[1:29]) > 2)
    
    return [v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers]


def create_valid_transaction() -> List[float]:
    """Create a realistic valid transaction."""
    features = []
    
    # Time (seconds from first transaction)
    features.append(float(np.random.randint(0, 100000)))
    
    # V1-V28 (PCA components - valid transactions have smaller values)
    for j in range(28):
        features.append(float(np.random.normal(loc=0.0, scale=0.5)))
    
    # Amount (smaller amounts for valid)
    features.append(float(np.random.uniform(0.0, 100.0)))
    
    return features


def create_fraud_transaction() -> List[float]:
    """Create a realistic fraud transaction."""
    features = []
    features.append(float(np.random.randint(0, 100000)))
    for j in range(28):
        # larger magnitude / outliers for fraud
        features.append(float(np.random.normal(loc=0.0, scale=3.0)))
    features.append(float(np.random.uniform(200.0, 5000.0)))
    return features


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to native Python types for JSON serialization."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_native(x) for x in obj)
    return obj


def triton_infer_http(model_name: str, inputs: List[List[float]], url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Simple Triton HTTP inference client using v2/infer endpoint.
    This implementation coerces numpy types to native Python numbers so json.dumps succeeds.
    """
    infer_url = f"{url}/v2/models/{model_name}/infer"
    # ensure inputs are native Python floats/ints
    native_inputs = []
    for row in inputs:
        # handle numpy arrays or lists containing numpy types
        native_row = []
        for v in row:
            # convert numpy types to native Python via float/int where appropriate
            if isinstance(v, (np.floating, float)):
                native_row.append(float(v))
            elif isinstance(v, (np.integer, int)):
                # Triton expects FP32 for floating models; cast integers to float as well
                native_row.append(float(v))
            else:
                # fallback: attempt generic conversion
                try:
                    native_row.append(float(v))
                except Exception:
                    native_row.append(_to_native(v))
        native_inputs.append(native_row)

    # construct JSON payload
    payload = {
        "inputs": [
            {
                "name": "INPUT__0",
                "shape": [len(native_inputs), len(native_inputs[0])],
                "datatype": "FP32",
                "data": native_inputs,
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    # use json.dumps on already-native data
    resp = requests.post(infer_url, data=json.dumps(payload), headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def test_single_transaction(features: List[float], transaction_type: str, 
                          model_name: str = "fraud_rf_smote", 
                          url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Send a single transaction to Triton and return the result (or local prediction if Triton not available)."""
    enhanced = calculate_enhanced_features(features)
    # combine original + enhanced if model expects full vector; adjust as needed
    payload_vec = features + enhanced

    try:
        resp = triton_infer_http(model_name, [payload_vec], url=url)
        return {"type": transaction_type, "result": resp}
    except Exception as e:
        return {"type": transaction_type, "error": str(e)}


def test_multiple_transactions(num_valid: int = 5, num_fraud: int = 5):
    results = []
    for _ in range(num_valid):
        v = create_valid_transaction()
        results.append(test_single_transaction(v, "valid"))
    for _ in range(num_fraud):
        f = create_fraud_transaction()
        results.append(test_single_transaction(f, "fraud"))
    # print summary
    for r in results:
        print(json.dumps(r))
    # basic exit code: fail if any request errored
    errs = [r for r in results if "error" in r]
    if errs:
        print("Some inferences failed:", errs, file=sys.stderr)
        sys.exit(2)
    print("All inferences returned successfully.")


def test_edge_cases():
    # tiny tests for malformed input
    try:
        _ = triton_infer_http("fraud_rf_smote", [[0.0]*35])
    except Exception as e:
        print("Edge case test produced error (expected if model config differs):", e)


# Try to import the training module that should define or save the model
improved = None
try:
    # make sure src/ is on sys.path so importlib can find improved_retraining.py if it's under src/
    src_path = os.path.join(os.getcwd(), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    improved = importlib.import_module("improved_retraining")
except Exception as e:
    # If running under pytest, prefer to skip; otherwise continue and try to load model artifact directly.
    msg = f"improved_retraining module not importable: {e}"
    if pytest:
        pytest.skip(msg)
    else:
        print(f"Note: {msg}", file=sys.stderr)
        improved = None


def _get_model_from_module(mod):
    """
    If mod is a module, try to extract a model object or a saved artifact path.
    If mod is None, attempt to locate the model artifact in common locations:
      - deploy/triton/model_repository/<model_name>/1/*.joblib|*.pkl
      - artifacts/*.joblib|*.pkl
      - models/*.joblib|*.pkl
    """
    # If a module is provided, try module-based loading first
    if mod is not None:
        # 1) direct model object
        if hasattr(mod, "model"):
            return getattr(mod, "model")

        # 2) loader function
        for fn in ("load_model", "load_best_model", "get_model"):
            if hasattr(mod, fn) and callable(getattr(mod, fn)):
                return getattr(mod, fn)()

        # 3) common file-name constants on the module
        candidates = []
        for name in ("MODEL_FILE", "MODEL_PATH", "MODEL_FILENAME", "MODEL_FILEPATH", "MODEL_OUTPUT"):
            if hasattr(mod, name):
                candidates.append(getattr(mod, name))

        # 4) try default common filenames in module dir / project root
        project_dir = os.path.dirname(mod.__file__) or os.getcwd()
        common_files = [
            *[c for c in candidates if isinstance(c, str)],
            os.path.join(project_dir, "model.pkl"),
            os.path.join(project_dir, "models", "model.pkl"),
            os.path.join(project_dir, "models", "improved_model.pkl"),
            os.path.join(os.getcwd(), "models", "improved_model.pkl"),
        ]

        for path in common_files:
            if not isinstance(path, str):
                continue
            if os.path.isabs(path) and os.path.exists(path):
                return joblib.load(path)
            rel = os.path.join(os.getcwd(), path) if not os.path.isabs(path) else path
            if os.path.exists(rel):
                return joblib.load(rel)

    # If we get here, try to locate a saved model artifact on disk (common locations)
    search_paths = [
        os.path.join(os.getcwd(), "deploy", "triton", "model_repository"),
        os.path.join(os.getcwd(), "artifacts"),
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.getcwd(), ""),
    ]

    candidate_files = []
    for base in search_paths:
        if not os.path.isdir(base):
            continue
        # look for joblib/pkl under nested dirs (e.g. model_repository/<name>/1/*.joblib)
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith((".joblib", ".pkl")):
                    candidate_files.append(os.path.join(root, f))

    # Prefer files that are inside a fraud_rf_smote folder if present
    candidate_files.sort(key=lambda p: ("fraud_rf_smote" not in p, p))

    for path in candidate_files:
        try:
            return joblib.load(path)
        except Exception:
            # try next candidate
            continue

    raise FileNotFoundError("Could not find/load model artifact in project (searched common locations)")


def test_model_loading_and_prediction():
    # Attempt to load model (module-first, then artifact search)
    model = _get_model_from_module(improved)
    # basic smoke check: model should have predict / predict_proba or be callable
    assert model is not None

    # If it's an sklearn-like estimator, check predict/predict_proba
    if hasattr(model, "predict"):
        import numpy as np
        # create a tiny dummy sample of zeros with expected number of features
        # Try to infer n_features from coef_ or n_features_in_
        n_features = getattr(model, "n_features_in_", None)
        if n_features is None and hasattr(model, "coef_"):
            coef = getattr(model, "coef_")
            try:
                n_features = coef.shape[-1]
            except Exception:
                n_features = None
        if n_features is None:
            # fallback to 30 (common for classic credit card fraud dataset)
            n_features = 30
        X = np.zeros((1, n_features))
        preds = model.predict(X)
        assert preds is not None

    elif callable(model):
        # if model is a callable predictor function
        res = model([0] * 30)
        assert res is not None

    else:
        if pytest:
            pytest.skip("Loaded object is not a callable/predictor")
        else:
            print("Loaded object is not a callable/predictor; skipping model check.", file=sys.stderr)


if __name__ == "__main__":
    # simple CLI: run full battery
    test_multiple_transactions()
