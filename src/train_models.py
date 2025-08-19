import argparse
import os
from typing import Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
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

def plot_and_save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str, filename: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)

def plot_and_save_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str, filename: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)

def plot_and_save_score_distribution(y_true: np.ndarray, y_proba: np.ndarray, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y_proba[y_true == 0], bins=50, color="#1f77b4", alpha=0.6, stat="density", label="Class 0", ax=ax)
    sns.histplot(y_proba[y_true == 1], bins=50, color="#d62728", alpha=0.6, stat="density", label="Class 1", ax=ax)
    ax.set_xlabel("Predicted probability of class 1")
    ax.set_title(title)
    ax.legend()
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

    # Additional prediction visualizations when probabilities are available
    if y_proba is not None:
        base_filename = f"{model_name.lower().replace(' ', '_')}_{suffix}"
        plot_and_save_roc_curve(
            y_test,
            y_proba,
            title=f"ROC Curve - {model_name} ({suffix})",
            filename=f"{base_filename}_roc.png",
        )
        plot_and_save_pr_curve(
            y_test,
            y_proba,
            title=f"Precision-Recall - {model_name} ({suffix})",
            filename=f"{base_filename}_pr.png",
        )
        plot_and_save_score_distribution(
            y_test,
            y_proba,
            title=f"Score Distribution - {model_name} ({suffix})",
            filename=f"{base_filename}_score_distribution.png",
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


