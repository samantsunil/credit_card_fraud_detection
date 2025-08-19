import argparse
import os
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from common import (
    ensure_dirs,
    PLOTS_DIR,
    MODELS_DIR,
    load_data,
    train_test_split_features,
    compute_metrics,
    save_model,
    pretty_print_classification_report,
)


def plot_and_save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def train_and_evaluate(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    suffix: str,
) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Probabilities for ROC-AUC when supported
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"\n== {model_name} ({suffix}) ==")
    print(pretty_print_classification_report(y_test, y_pred))
    print("Metrics:", metrics)

    plot_and_save_confusion_matrix(
        y_test,
        y_pred,
        title=f"{model_name} ({suffix})",
        filename=f"{model_name.lower().replace(' ', '_')}_{suffix}_confusion_matrix.png",
    )

    # Save model
    save_model(
        model,
        os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_{suffix}.joblib"),
    )

    metrics["model"] = model_name
    metrics["setting"] = suffix
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DT and RF (baseline and SMOTE)")
    parser.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    args = parser.parse_args()

    ensure_dirs()

    df = load_data(args.data_path)

    # Baseline split
    X_train, X_test, y_train, y_test = train_test_split_features(
        df, label_column="Class", test_size=args.test_size, random_state=args.random_state
    )

    # Models
    dt = DecisionTreeClassifier(random_state=args.random_state)
    rf = RandomForestClassifier(n_estimators=200, random_state=args.random_state, n_jobs=-1)

    # Baseline (no resampling)
    baseline_metrics = []
    baseline_metrics.append(
        train_and_evaluate(dt, X_train, X_test, y_train, y_test, "Decision Tree", "baseline")
    )
    baseline_metrics.append(
        train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest", "baseline")
    )

    baseline_df = pd.DataFrame(baseline_metrics)
    baseline_df.to_csv(os.path.join("artifacts", "metrics_baseline.csv"), index=False)

    # SMOTE on training split only
    smote = SMOTE(random_state=args.random_state)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Reinitialize models for fair comparison
    dt_sm = DecisionTreeClassifier(random_state=args.random_state)
    rf_sm = RandomForestClassifier(n_estimators=200, random_state=args.random_state, n_jobs=-1)

    smote_metrics = []
    smote_metrics.append(
        train_and_evaluate(dt_sm, X_train_sm, X_test, y_train_sm, y_test, "Decision Tree", "smote")
    )
    smote_metrics.append(
        train_and_evaluate(rf_sm, X_train_sm, X_test, y_train_sm, y_test, "Random Forest", "smote")
    )

    smote_df = pd.DataFrame(smote_metrics)
    smote_df.to_csv(os.path.join("artifacts", "metrics_smote.csv"), index=False)

    # Combined view
    combined = pd.concat([baseline_df, smote_df], ignore_index=True)
    print("\nCombined metrics:\n", combined)


if __name__ == "__main__":
    main()


