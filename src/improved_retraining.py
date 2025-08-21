#!/usr/bin/env python3
"""
Improved retraining script for Credit Card Fraud Detection
Uses advanced techniques for handling class imbalance and fraud detection
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
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


def create_fraud_features(X: np.ndarray) -> np.ndarray:
    """Create additional fraud-specific features."""
    print("🔧 Creating fraud-specific features...")
    
    # Create new features that might help with fraud detection
    new_features = []
    
    for i in range(X.shape[0]):
        features = X[i]
        
        # Feature 1: Sum of absolute values of V1-V28 (indicates overall magnitude)
        v_sum_abs = np.sum(np.abs(features[1:29]))
        
        # Feature 2: Standard deviation of V1-V28 (indicates variability)
        v_std = np.std(features[1:29])
        
        # Feature 3: Maximum absolute value of V1-V28
        v_max_abs = np.max(np.abs(features[1:29]))
        
        # Feature 4: Amount to time ratio (fraud often happens at unusual times)
        amount_time_ratio = features[29] / (features[0] + 1)  # +1 to avoid division by zero
        
        # Feature 5: Number of V features with absolute value > 2 (outliers)
        v_outliers = np.sum(np.abs(features[1:29]) > 2)
        
        new_features.append([v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers])
    
    # Combine original features with new features
    X_enhanced = np.hstack([X, np.array(new_features)])
    
    print(f"Enhanced features shape: {X_enhanced.shape}")
    return X_enhanced


def train_with_simple_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """Train model with simple SMOTE approach - resample entire dataset first, then split."""
    print("\n🔄 Training with simple SMOTE approach...")
    
    # Apply SMOTE to entire dataset first
    print("Applying SMOTE to entire dataset...")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Resampled dataset shape: {X_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y)}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")
    
    # Split the resampled data (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=random_state, stratify=y_resampled
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Create Random Forest model
    print("\n🌲 Training Random Forest model...")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train on resampled data
    rf.fit(X_train, y_train)
    
    return rf, X_train, X_test, y_train, y_test, X_resampled, y_resampled


def evaluate_model_comprehensive(model, X_test: np.ndarray, y_test: np.ndarray, 
                               X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Comprehensive model evaluation with detailed analysis."""
    print("\n📊 Comprehensive model evaluation...")
    
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
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\n🔍 Top 10 Feature Importances:")
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        for i, feature_idx in enumerate(top_features):
            print(f"Feature {feature_idx}: {feature_importance[feature_idx]:.4f}")
    
    return {
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'y_pred_test': y_pred_test,
        'y_proba_test': y_proba_test,
        'y_pred_train': y_pred_train,
        'y_proba_train': y_proba_train
    }


def create_realistic_test_data() -> tuple:
    """Create more realistic test data based on actual fraud patterns."""
    print("\n🧪 Creating realistic validation test data...")
    
    # Valid transaction (based on typical legitimate patterns)
    valid_features = [
        50000.0,  # Time - normal transaction time
        -0.05, 0.08, -0.12, 0.06, -0.09, 0.11, -0.07, 0.05,  # V1-V8 - small values
        -0.04, 0.09, -0.13, 0.07, -0.08, 0.10, -0.06, 0.04,  # V9-V16 - small values
        -0.05, 0.08, -0.11, 0.06, -0.09, 0.12, -0.07, 0.05,  # V17-V24 - small values
        -0.04, 0.09, -0.12, 0.07,  # V25-V28 - small values
        25.0  # Amount - normal transaction amount
    ]
    
    # Add enhanced features for valid transaction
    v_sum_abs = np.sum(np.abs(valid_features[1:29]))
    v_std = np.std(valid_features[1:29])
    v_max_abs = np.max(np.abs(valid_features[1:29]))
    amount_time_ratio = valid_features[29] / (valid_features[0] + 1)
    v_outliers = np.sum(np.abs(valid_features[1:29]) > 2)
    
    valid_features.extend([v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers])
    
    # Fraud transaction (based on typical fraud patterns)
    fraud_features = [
        120000.0,  # Time - unusual transaction time
        3.2, -4.1, 5.8, -3.9, 4.7, -5.2, 3.8, -4.5,  # V1-V8 - large values
        4.1, -5.8, 3.9, -4.7, 5.2, -3.8, 4.5, -5.1,  # V9-V16 - large values
        3.7, -4.9, 5.4, -3.6, 4.8, -5.3, 3.9, -4.2,  # V17-V24 - large values
        5.1, -3.7, 4.6, -5.4,  # V25-V28 - large values
        1500.0  # Amount - large transaction amount
    ]
    
    # Add enhanced features for fraud transaction
    v_sum_abs = np.sum(np.abs(fraud_features[1:29]))
    v_std = np.std(fraud_features[1:29])
    v_max_abs = np.max(np.abs(fraud_features[1:29]))
    amount_time_ratio = fraud_features[29] / (fraud_features[0] + 1)
    v_outliers = np.sum(np.abs(fraud_features[1:29]) > 2)
    
    fraud_features.extend([v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers])
    
    return valid_features, fraud_features


def validate_model_thoroughly(model, valid_features: list, fraud_features: list):
    """Thorough model validation with multiple test cases."""
    print("\n🔍 Thorough model validation...")
    
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
    
    # Test multiple variations
    print("\n🧪 Testing multiple variations...")
    
    # Test with different amounts
    for amount in [10.0, 50.0, 100.0, 500.0, 1000.0]:
        test_features = valid_features.copy()
        test_features[29] = amount  # Update amount
        # Recalculate enhanced features
        v_sum_abs = np.sum(np.abs(test_features[1:29]))
        v_std = np.std(test_features[1:29])
        v_max_abs = np.max(np.abs(test_features[1:29]))
        amount_time_ratio = test_features[29] / (test_features[0] + 1)
        v_outliers = np.sum(np.abs(test_features[1:29]) > 2)
        test_features[30:] = [v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers]
        
        pred = model.predict([test_features])[0]
        proba = model.predict_proba([test_features])[0][1]
        print(f"Amount ${amount}: Pred={pred}, Prob={proba:.4f}")
    
    # Test with different V-feature magnitudes
    print("\n🧪 Testing with different V-feature magnitudes...")
    for multiplier in [0.1, 0.5, 1.0, 2.0, 5.0]:
        test_features = fraud_features.copy()
        # Scale V1-V28 features
        for i in range(1, 29):
            test_features[i] *= multiplier
        # Recalculate enhanced features
        v_sum_abs = np.sum(np.abs(test_features[1:29]))
        v_std = np.std(test_features[1:29])
        v_max_abs = np.max(np.abs(test_features[1:29]))
        amount_time_ratio = test_features[29] / (test_features[0] + 1)
        v_outliers = np.sum(np.abs(test_features[1:29]) > 2)
        test_features[30:] = [v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers]
        
        pred = model.predict([test_features])[0]
        proba = model.predict_proba([test_features])[0][1]
        print(f"V-magnitude x{multiplier}: Pred={pred}, Prob={proba:.4f}")
    
    # Check if predictions make sense
    if valid_pred == 0 and fraud_pred == 1:
        print("\n✅ Model predictions are correct!")
        print("The model can distinguish between valid and fraud transactions.")
    else:
        print("\n❌ Model predictions are incorrect!")
        print("This suggests the model needs further improvement.")
        print("Consider:")
        print("1. Using different resampling techniques")
        print("2. Feature engineering")
        print("3. Hyperparameter tuning")
        print("4. Using ensemble methods")


def save_model_and_artifacts(model, evaluation_results: dict, output_dir: str = "artifacts"):
    """Save model and evaluation artifacts."""
    print(f"\n💾 Saving model and artifacts to {output_dir}...")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/models/random_forest_improved.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save evaluation results
    results_path = f"{output_dir}/improved_evaluation_results.csv"
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


def deploy_to_triton(model_path: str, triton_model_dir: str = "deploy/triton/model_repository/fraud_rf_smote/1"):
    """Deploy the improved model to Triton."""
    print(f"\n🚀 Deploying improved model to Triton...")
    
    # Create Triton model directory
    os.makedirs(triton_model_dir, exist_ok=True)
    
    # Copy model to Triton directory
    import shutil
    triton_model_path = f"{triton_model_dir}/random_forest_smote.joblib"
    shutil.copy2(model_path, triton_model_path)
    print(f"Model copied to: {triton_model_path}")
    
    print("✅ Improved model deployed to Triton!")
    print("Restart your Triton server to load the new model.")


def main():
    parser = argparse.ArgumentParser(description="Improved Credit Card Fraud Detection Model Training")
    parser.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Triton after training")
    
    args = parser.parse_args()
    
    print("🚀 Credit Card Fraud Detection - Improved Model Training")
    print("=" * 70)
    
    # Load and prepare data
    X, y = load_and_prepare_data(args.data_path)
    
    # Create enhanced features
    X_enhanced = create_fraud_features(X)
    
    # Train model with simple SMOTE
    model, X_train, X_test, y_train, y_test, X_resampled, y_resampled = train_with_simple_smote(
        X_enhanced, y, args.random_state
    )
    
    # Evaluate model
    evaluation_results = evaluate_model_comprehensive(model, X_test, y_test, X_train, y_train)
    
    # Save model and artifacts
    model_path = save_model_and_artifacts(model, evaluation_results)
    
    # Create and validate test data
    valid_features, fraud_features = create_realistic_test_data()
    validate_model_thoroughly(model, valid_features, fraud_features)
    
    # Deploy to Triton if requested
    if args.deploy:
        deploy_to_triton(model_path)
    
    print("\n✅ Improved training completed!")
    print(f"Model saved to: {model_path}")
    
    if args.deploy:
        print("Improved model deployed to Triton. Restart your Triton server to use the new model.")


if __name__ == "__main__":
    main()
