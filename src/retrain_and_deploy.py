#!/usr/bin/env python3
"""
Comprehensive retraining script for Credit Card Fraud Detection
Ensures proper SMOTE application and model evaluation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib backend
import matplotlib
matplotlib.use("Agg")


def load_and_prepare_data(data_path: str) -> tuple:
    """Load and prepare the dataset."""
    print("📊 Loading dataset...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    print(f"Class distribution (%):\n{df['Class'].value_counts(normalize=True) * 100}")
    
    # Separate features and target
    X = df.drop(columns=['Class']).values
    y = df['Class'].values
    
    return X, y


def train_with_proper_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """Train model with proper SMOTE application."""
    print("\n🔄 Training with SMOTE...")
    
    # Split data first (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Apply SMOTE only to training data
    print("\n📈 Applying SMOTE to training data...")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Training set shape: {X_train_smote.shape}")
    print(f"After SMOTE - Training class distribution: {np.bincount(y_train_smote)}")
    
    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Train on SMOTE data
    rf.fit(X_train_smote, y_train_smote)
    
    return rf, X_train, X_test, y_train, y_test, X_train_smote, y_train_smote


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Comprehensive model evaluation."""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_proba_train = model.predict_proba(X_train)[:, 1]
    
    # Test set metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_test)
    }
    
    # Training set metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_pred_train),
        'precision': precision_score(y_train, y_pred_train, zero_division=0),
        'recall': recall_score(y_train, y_pred_train, zero_division=0),
        'f1': f1_score(y_train, y_pred_train, zero_division=0),
        'roc_auc': roc_auc_score(y_train, y_proba_train)
    }
    
    print("\n📈 Test Set Performance:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    print("\n📈 Training Set Performance:")
    print(f"Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall: {train_metrics['recall']:.4f}")
    print(f"F1-Score: {train_metrics['f1']:.4f}")
    print(f"ROC-AUC: {train_metrics['roc_auc']:.4f}")
    
    # Confusion matrices
    print("\n🔍 Test Set Confusion Matrix:")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)
    
    print("\n🔍 Training Set Confusion Matrix:")
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(cm_train)
    
    # Classification reports
    print("\n📋 Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test, digits=4))
    
    print("\n📋 Training Set Classification Report:")
    print(classification_report(y_train, y_pred_train, digits=4))
    
    return {
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'y_pred_test': y_pred_test,
        'y_proba_test': y_proba_test,
        'y_pred_train': y_pred_train,
        'y_proba_train': y_proba_train
    }


def save_model_and_artifacts(model, evaluation_results: dict, output_dir: str = "artifacts"):
    """Save model and evaluation artifacts."""
    print(f"\n💾 Saving model and artifacts to {output_dir}...")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/models/random_forest_smote_retrained.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save evaluation results
    results_path = f"{output_dir}/retraining_evaluation_results.csv"
    results_df = pd.DataFrame({
        'Dataset': ['Test', 'Train'],
        'Accuracy': [evaluation_results['test_metrics']['accuracy'], 
                    evaluation_results['train_metrics']['accuracy']],
        'Precision': [evaluation_results['test_metrics']['precision'], 
                     evaluation_results['train_metrics']['precision']],
        'Recall': [evaluation_results['test_metrics']['recall'], 
                  evaluation_results['train_metrics']['recall']],
        'F1_Score': [evaluation_results['test_metrics']['f1'], 
                    evaluation_results['train_metrics']['f1']],
        'ROC_AUC': [evaluation_results['test_metrics']['roc_auc'], 
                   evaluation_results['train_metrics']['roc_auc']]
    })
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved to: {results_path}")
    
    return model_path


def create_test_data_for_validation() -> tuple:
    """Create realistic test data for validation."""
    print("\n🧪 Creating validation test data...")
    
    # Valid transaction (should be predicted as 0)
    valid_features = [
        1000.0,  # Time
        -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2,  # V1-V8
        -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2,  # V9-V16
        -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2,  # V17-V24
        -0.1, 0.2, -0.3, 0.1,  # V25-V28
        50.0  # Amount
    ]
    
    # Fraud transaction (should be predicted as 1)
    fraud_features = [
        2000.0,  # Time
        2.5, -3.1, 4.2, -2.8, 3.5, -4.1, 2.9, -3.7,  # V1-V8
        3.2, -4.5, 2.8, -3.9, 4.1, -2.7, 3.6, -4.2,  # V9-V16
        2.9, -3.8, 4.3, -2.6, 3.7, -4.4, 2.5, -3.3,  # V17-V24
        4.1, -2.9, 3.8, -4.6,  # V25-V28
        500.0  # Amount
    ]
    
    return valid_features, fraud_features


def validate_model(model, valid_features: list, fraud_features: list):
    """Validate model with known test cases."""
    print("\n🔍 Validating model with test cases...")
    
    # Test valid transaction
    valid_pred = model.predict([valid_features])[0]
    valid_proba = model.predict_proba([valid_features])[0][1]
    print(f"Valid transaction - Prediction: {valid_pred} (0=legitimate, 1=fraud)")
    print(f"Valid transaction - Fraud probability: {valid_proba:.4f}")
    
    # Test fraud transaction
    fraud_pred = model.predict([fraud_features])[0]
    fraud_proba = model.predict_proba([fraud_features])[0][1]
    print(f"Fraud transaction - Prediction: {fraud_pred} (0=legitimate, 1=fraud)")
    print(f"Fraud transaction - Fraud probability: {fraud_proba:.4f}")
    
    # Check if predictions make sense
    if valid_pred == 0 and fraud_pred == 1:
        print("✅ Model predictions are correct!")
    else:
        print("❌ Model predictions are incorrect!")
        print("This suggests the model needs improvement.")


def deploy_to_triton(model_path: str, triton_model_dir: str = "deploy/triton/model_repository/fraud_rf_smote/1"):
    """Deploy the retrained model to Triton."""
    print(f"\n🚀 Deploying model to Triton...")
    
    # Create Triton model directory
    os.makedirs(triton_model_dir, exist_ok=True)
    
    # Copy model to Triton directory
    import shutil
    triton_model_path = f"{triton_model_dir}/random_forest_smote.joblib"
    shutil.copy2(model_path, triton_model_path)
    print(f"Model copied to: {triton_model_path}")
    
    print("✅ Model deployed to Triton!")
    print("Restart your Triton server to load the new model.")


def main():
    parser = argparse.ArgumentParser(description="Retrain Credit Card Fraud Detection Model")
    parser.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Triton after training")
    
    args = parser.parse_args()
    
    print("🚀 Credit Card Fraud Detection - Model Retraining")
    print("=" * 60)
    
    # Load and prepare data
    X, y = load_and_prepare_data(args.data_path)
    
    # Train model with proper SMOTE
    model, X_train, X_test, y_train, y_test, X_train_smote, y_train_smote = train_with_proper_smote(
        X, y, args.random_state
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Save model and artifacts
    model_path = save_model_and_artifacts(model, evaluation_results)
    
    # Create and validate test data
    valid_features, fraud_features = create_test_data_for_validation()
    validate_model(model, valid_features, fraud_features)
    
    # Deploy to Triton if requested
    if args.deploy:
        deploy_to_triton(model_path)
    
    print("\n✅ Retraining completed!")
    print(f"Model saved to: {model_path}")
    
    if args.deploy:
        print("Model deployed to Triton. Restart your Triton server to use the new model.")


if __name__ == "__main__":
    main()
