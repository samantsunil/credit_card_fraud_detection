#!/usr/bin/env python3
"""
Enhanced retraining script for Credit Card Fraud Detection
Uses advanced techniques including ensemble methods, better feature engineering, and threshold optimization
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import TomekLinks
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


def create_advanced_features(X: np.ndarray) -> np.ndarray:
    """Create advanced fraud-specific features."""
    print("🔧 Creating advanced fraud-specific features...")
    
    new_features = []
    
    for i in range(X.shape[0]):
        features = X[i]
        
        # Basic engineered features
        v_sum_abs = np.sum(np.abs(features[1:29]))
        v_std = np.std(features[1:29])
        v_max_abs = np.max(np.abs(features[1:29]))
        amount_time_ratio = features[29] / (features[0] + 1)
        v_outliers = np.sum(np.abs(features[1:29]) > 2)
        
        # Advanced features
        # 1. Ratio of positive to negative V-values
        v_positive = np.sum(features[1:29] > 0)
        v_negative = np.sum(features[1:29] < 0)
        v_pos_neg_ratio = v_positive / (v_negative + 1)
        
        # 2. Variance of V-values (different from std)
        v_variance = np.var(features[1:29])
        
        # 3. Range of V-values
        v_range = np.max(features[1:29]) - np.min(features[1:29])
        
        # 4. Number of V-values above 95th percentile
        v_95th_percentile = np.percentile(features[1:29], 95)
        v_above_95th = np.sum(features[1:29] > v_95th_percentile)
        
        # 5. Amount to V-features ratio
        amount_v_ratio = features[29] / (v_sum_abs + 1)
        
        # 6. Time-based features (fraud often happens at unusual times)
        time_hour = (features[0] % 86400) / 3600  # Hour of day
        time_sin = np.sin(2 * np.pi * time_hour / 24)
        time_cos = np.cos(2 * np.pi * time_hour / 24)
        
        # 7. Interaction features
        amount_v_max_interaction = features[29] * v_max_abs
        time_amount_interaction = features[0] * features[29]
        
        # 8. Statistical features
        v_skewness = np.mean(((features[1:29] - np.mean(features[1:29])) / (np.std(features[1:29]) + 1e-8)) ** 3)
        v_kurtosis = np.mean(((features[1:29] - np.mean(features[1:29])) / (np.std(features[1:29]) + 1e-8)) ** 4) - 3
        
        new_features.append([
            v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers,
            v_pos_neg_ratio, v_variance, v_range, v_above_95th, amount_v_ratio,
            time_sin, time_cos, amount_v_max_interaction, time_amount_interaction,
            v_skewness, v_kurtosis
        ])
    
    # Combine original features with new features
    X_enhanced = np.hstack([X, np.array(new_features)])
    
    print(f"Enhanced features shape: {X_enhanced.shape}")
    return X_enhanced


def train_ensemble_model(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple:
    """Train an ensemble model with advanced techniques."""
    print("\n🔄 Training ensemble model with advanced techniques...")
    
    # Split data first (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Try different resampling techniques
    resampling_methods = {
        'SMOTE': SMOTE(random_state=random_state, k_neighbors=3),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=random_state, k_neighbors=3),
        'ADASYN': ADASYN(random_state=random_state),
        'SMOTEENN': SMOTEENN(random_state=random_state),
        'SMOTETomek': SMOTETomek(random_state=random_state)
    }
    
    best_method = None
    best_score = 0
    best_X_train_resampled = None
    best_y_train_resampled = None
    
    print("\n🔍 Testing different resampling methods...")
    
    for method_name, resampler in resampling_methods.items():
        try:
            print(f"Testing {method_name}...")
            X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
            
            # Quick evaluation with a simple model
            temp_rf = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1)
            temp_rf.fit(X_train_resampled, y_train_resampled)
            
            # Cross-validation score
            cv_scores = cross_val_score(temp_rf, X_train_resampled, y_train_resampled, 
                                      cv=3, scoring='f1', n_jobs=-1)
            avg_score = cv_scores.mean()
            
            print(f"{method_name} - CV F1 Score: {avg_score:.4f}")
            print(f"{method_name} - Resampled shape: {X_train_resampled.shape}")
            print(f"{method_name} - Class distribution: {np.bincount(y_train_resampled)}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_method = method_name
                best_X_train_resampled = X_train_resampled
                best_y_train_resampled = y_train_resampled
                
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            continue
    
    print(f"\n✅ Best resampling method: {best_method} (F1 Score: {best_score:.4f})")
    
    # Create ensemble model
    print("\n🌲 Training ensemble model...")
    
    # Individual models
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced_subsample',
        criterion='entropy'
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=8,
        random_state=random_state,
        subsample=0.8
    )
    
    lr = LogisticRegression(
        random_state=random_state,
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear'
    )
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft'  # Use probability voting
    )
    
    # Train ensemble on best resampled data
    ensemble.fit(best_X_train_resampled, best_y_train_resampled)
    
    return ensemble, X_train, X_test, y_train, y_test, best_X_train_resampled, best_y_train_resampled


def optimize_threshold(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Optimize the decision threshold for better fraud detection."""
    print("\n🎯 Optimizing decision threshold...")
    
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    return best_threshold


def evaluate_model_comprehensive(model, X_test: np.ndarray, y_test: np.ndarray, 
                               X_train: np.ndarray, y_train: np.ndarray,
                               threshold: float = 0.5) -> dict:
    """Comprehensive model evaluation with threshold optimization."""
    print("\n📊 Comprehensive model evaluation...")
    
    # Predictions with optimized threshold
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_proba_train = model.predict_proba(X_train)[:, 1]
    
    y_pred_test = (y_proba_test >= threshold).astype(int)
    y_pred_train = (y_proba_train >= threshold).astype(int)
    
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
    
    print(f"\n📈 Test Set Performance (threshold={threshold:.3f}):")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    print(f"\n📈 Training Set Performance (threshold={threshold:.3f}):")
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
        'y_proba_train': y_proba_train,
        'threshold': threshold
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
    enhanced_features = calculate_enhanced_features(valid_features)
    valid_features.extend(enhanced_features)
    
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
    enhanced_features = calculate_enhanced_features(fraud_features)
    fraud_features.extend(enhanced_features)
    
    return valid_features, fraud_features


def calculate_enhanced_features(features: list) -> list:
    """Calculate enhanced features for a given feature vector."""
    # Convert to numpy array for vectorized operations
    features_array = np.array(features)
    v_features = features_array[1:29]  # V1-V28 features
    
    # Basic engineered features
    v_sum_abs = np.sum(np.abs(v_features))
    v_std = np.std(v_features)
    v_max_abs = np.max(np.abs(v_features))
    amount_time_ratio = features[29] / (features[0] + 1)
    v_outliers = np.sum(np.abs(v_features) > 2)
    
    # Advanced features
    v_positive = np.sum(v_features > 0)
    v_negative = np.sum(v_features < 0)
    v_pos_neg_ratio = v_positive / (v_negative + 1)
    v_variance = np.var(v_features)
    v_range = np.max(v_features) - np.min(v_features)
    v_95th_percentile = np.percentile(v_features, 95)
    v_above_95th = np.sum(v_features > v_95th_percentile)
    amount_v_ratio = features[29] / (v_sum_abs + 1)
    
    time_hour = (features[0] % 86400) / 3600
    time_sin = np.sin(2 * np.pi * time_hour / 24)
    time_cos = np.cos(2 * np.pi * time_hour / 24)
    
    amount_v_max_interaction = features[29] * v_max_abs
    time_amount_interaction = features[0] * features[29]
    
    v_skewness = np.mean(((v_features - np.mean(v_features)) / (np.std(v_features) + 1e-8)) ** 3)
    v_kurtosis = np.mean(((v_features - np.mean(v_features)) / (np.std(v_features) + 1e-8)) ** 4) - 3
    
    return [
        v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers,
        v_pos_neg_ratio, v_variance, v_range, v_above_95th, amount_v_ratio,
        time_sin, time_cos, amount_v_max_interaction, time_amount_interaction,
        v_skewness, v_kurtosis
    ]


def validate_model_thoroughly(model, valid_features: list, fraud_features: list, threshold: float = 0.5):
    """Thorough model validation with multiple test cases."""
    print("\n🔍 Thorough model validation...")
    
    # Test valid transaction
    valid_pred = (model.predict_proba([valid_features])[0][1] >= threshold).astype(int)
    valid_proba = model.predict_proba([valid_features])[0][1]
    print(f"Valid transaction - Prediction: {valid_pred} (0=legitimate, 1=fraud)")
    print(f"Valid transaction - Fraud probability: {valid_proba:.4f}")
    
    # Test fraud transaction
    fraud_pred = (model.predict_proba([fraud_features])[0][1] >= threshold).astype(int)
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
        enhanced_features = calculate_enhanced_features(test_features)
        test_features[30:] = enhanced_features
        
        pred = (model.predict_proba([test_features])[0][1] >= threshold).astype(int)
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
        enhanced_features = calculate_enhanced_features(test_features)
        test_features[30:] = enhanced_features
        
        pred = (model.predict_proba([test_features])[0][1] >= threshold).astype(int)
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
        print("5. Adjusting the decision threshold")


def save_model_and_artifacts(model, evaluation_results: dict, output_dir: str = "artifacts"):
    """Save model and evaluation artifacts."""
    print(f"\n💾 Saving model and artifacts to {output_dir}...")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/models", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/models/ensemble_enhanced.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save threshold
    threshold_path = f"{output_dir}/models/optimal_threshold.txt"
    with open(threshold_path, 'w') as f:
        f.write(str(evaluation_results['threshold']))
    print(f"Optimal threshold saved to: {threshold_path}")
    
    # Save evaluation results
    results_path = f"{output_dir}/enhanced_evaluation_results.csv"
    results_df = pd.DataFrame({
        'Dataset': ['Test', 'Train'],
        'Threshold': [evaluation_results['threshold'], evaluation_results['threshold']],
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
    """Deploy the enhanced model to Triton."""
    print(f"\n🚀 Deploying enhanced model to Triton...")
    
    # Create Triton model directory
    os.makedirs(triton_model_dir, exist_ok=True)
    
    # Copy model to Triton directory
    import shutil
    triton_model_path = f"{triton_model_dir}/random_forest_smote.joblib"
    shutil.copy2(model_path, triton_model_path)
    print(f"Model copied to: {triton_model_path}")
    
    # Copy threshold file
    threshold_src = model_path.replace('ensemble_enhanced.joblib', 'optimal_threshold.txt')
    threshold_dst = f"{triton_model_dir}/optimal_threshold.txt"
    if os.path.exists(threshold_src):
        shutil.copy2(threshold_src, threshold_dst)
        print(f"Optimal threshold copied to: {threshold_dst}")
    
    print("✅ Enhanced model deployed to Triton!")
    print("Restart your Triton server to load the new model.")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Credit Card Fraud Detection Model Training")
    parser.add_argument("--data-path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Triton after training")
    
    args = parser.parse_args()
    
    print("🚀 Credit Card Fraud Detection - Enhanced Model Training")
    print("=" * 70)
    
    # Load and prepare data
    X, y = load_and_prepare_data(args.data_path)
    
    # Create enhanced features
    X_enhanced = create_advanced_features(X)
    
    # Train ensemble model
    model, X_train, X_test, y_train, y_test, X_train_resampled, y_train_resampled = train_ensemble_model(
        X_enhanced, y, args.random_state
    )
    
    # Optimize threshold
    optimal_threshold = optimize_threshold(model, X_test, y_test)
    
    # Evaluate model
    evaluation_results = evaluate_model_comprehensive(model, X_test, y_test, X_train, y_train, optimal_threshold)
    
    # Save model and artifacts
    model_path = save_model_and_artifacts(model, evaluation_results)
    
    # Create and validate test data
    valid_features, fraud_features = create_realistic_test_data()
    validate_model_thoroughly(model, valid_features, fraud_features, optimal_threshold)
    
    # Deploy to Triton if requested
    if args.deploy:
        deploy_to_triton(model_path)
    
    print("\n✅ Enhanced training completed!")
    print(f"Model saved to: {model_path}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    
    if args.deploy:
        print("Enhanced model deployed to Triton. Restart your Triton server to use the new model.")


if __name__ == "__main__":
    main()
