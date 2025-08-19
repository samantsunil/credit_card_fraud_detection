import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


ARTIFACTS_DIR = os.path.join("artifacts")
EDA_DIR = os.path.join(ARTIFACTS_DIR, "eda")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")


def ensure_dirs() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def train_test_split_features(
    df: pd.DataFrame,
    label_column: str = "Class",
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop(columns=[label_column]).values
    y = df[label_column].values
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        # fallback if probabilities not available
        metrics["roc_auc"] = np.nan
    return metrics


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def pretty_print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, digits=4, zero_division=0)


