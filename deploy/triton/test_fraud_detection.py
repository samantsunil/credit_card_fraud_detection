#!/usr/bin/env python3
"""
Comprehensive test for Credit Card Fraud Detection with Triton
Tests both valid and fraud transaction patterns
"""

import json
import numpy as np
import requests
from typing import List, Dict, Any


def calculate_enhanced_features(features: list) -> list:
    """Calculate enhanced features for a given feature vector."""
    # Basic engineered features
    v_sum_abs = np.sum(np.abs(features[1:29]))
    v_std = np.std(features[1:29])
    v_max_abs = np.max(np.abs(features[1:29]))
    amount_time_ratio = features[29] / (features[0] + 1)
    v_outliers = np.sum(np.abs(features[1:29]) > 2)
    
    # Advanced features
    v_positive = np.sum(features[1:29] > 0)
    v_negative = np.sum(features[1:29] < 0)
    v_pos_neg_ratio = v_positive / (v_negative + 1)
    v_variance = np.var(features[1:29])
    v_range = np.max(features[1:29]) - np.min(features[1:29])
    v_95th_percentile = np.percentile(features[1:29], 95)
    v_above_95th = np.sum(features[1:29] > v_95th_percentile)
    amount_v_ratio = features[29] / (v_sum_abs + 1)
    
    time_hour = (features[0] % 86400) / 3600
    time_sin = np.sin(2 * np.pi * time_hour / 24)
    time_cos = np.cos(2 * np.pi * time_hour / 24)
    
    amount_v_max_interaction = features[29] * v_max_abs
    time_amount_interaction = features[0] * features[29]
    
    v_skewness = np.mean(((features[1:29] - np.mean(features[1:29])) / (np.std(features[1:29]) + 1e-8)) ** 3)
    v_kurtosis = np.mean(((features[1:29] - np.mean(features[1:29])) / (np.std(features[1:29]) + 1e-8)) ** 4) - 3
    
    return [
        v_sum_abs, v_std, v_max_abs, amount_time_ratio, v_outliers,
        v_pos_neg_ratio, v_variance, v_range, v_above_95th, amount_v_ratio,
        time_sin, time_cos, amount_v_max_interaction, time_amount_interaction,
        v_skewness, v_kurtosis
    ]


def create_valid_transaction() -> List[float]:
    """Create a realistic valid transaction."""
    features = []
    
    # Time (seconds from first transaction)
    features.append(float(np.random.randint(0, 100000)))
    
    # V1-V28 (PCA components - valid transactions have smaller values)
    for j in range(28):
        # Valid transactions typically have smaller absolute values
        features.append(np.random.normal(0, 0.5))
    
    # Amount (valid transactions typically have moderate amounts)
    features.append(abs(np.random.normal(50, 30)))
    
    # Add enhanced features for valid transaction
    enhanced_features = calculate_enhanced_features(features)
    features.extend(enhanced_features)
    
    return features


def create_fraud_transaction() -> List[float]:
    """Create a realistic fraud transaction."""
    features = []
    
    # Time (seconds from first transaction)
    features.append(float(np.random.randint(0, 100000)))
    
    # V1-V28 (PCA components - fraud transactions have larger values)
    for j in range(28):
        # Fraud transactions typically have larger absolute values
        features.append(np.random.normal(0, 2.0))
    
    # Amount (fraud transactions often have larger amounts)
    features.append(abs(np.random.normal(200, 100)))
    
    # Add enhanced features for fraud transaction
    enhanced_features = calculate_enhanced_features(features)
    features.extend(enhanced_features)
    
    return features


def test_single_transaction(features: List[float], transaction_type: str, 
                          model_name: str = "fraud_rf_smote", 
                          url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Test a single transaction."""
    
    request_data = {
        "inputs": [
            {
                "name": "INPUT__0",
                "datatype": "FP32",
                "shape": [1, 46],
                "data": features
            }
        ],
        "outputs": [
            {"name": "OUTPUT__PROB"},
            {"name": "OUTPUT__CLASS"}
        ]
    }
    
    try:
        response = requests.post(
            f"{url}/v2/models/{model_name}/infer",
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract results
            outputs = result.get("outputs", [])
            prob_output = None
            class_output = None
            
            for output in outputs:
                if output["name"] == "OUTPUT__PROB":
                    prob_output = output["data"][0]
                elif output["name"] == "OUTPUT__CLASS":
                    class_output = output["data"][0]
            
            fraud_status = "FRAUD" if class_output == 1 else "LEGITIMATE"
            
            return {
                "type": transaction_type,
                "probability": prob_output,
                "prediction": fraud_status,
                "correct": (transaction_type == "FRAUD" and class_output == 1) or 
                          (transaction_type == "VALID" and class_output == 0)
            }
        else:
            return {
                "type": transaction_type,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except Exception as e:
        return {
            "type": transaction_type,
            "error": str(e)
        }


def test_multiple_transactions(num_valid: int = 5, num_fraud: int = 5) -> None:
    """Test multiple valid and fraud transactions."""
    
    print("🚀 Testing Credit Card Fraud Detection Model")
    print("=" * 60)
    
    results = []
    
    # Test valid transactions
    print(f"\n📊 Testing {num_valid} VALID transactions...")
    for i in range(num_valid):
        features = create_valid_transaction()
        result = test_single_transaction(features, "VALID")
        results.append(result)
        
        if "error" not in result:
            print(f"  Valid {i+1}: Prob={result['probability']:.4f} | Pred={result['prediction']} | {'✅' if result['correct'] else '❌'}")
        else:
            print(f"  Valid {i+1}: ❌ {result['error']}")
    
    # Test fraud transactions
    print(f"\n📊 Testing {num_fraud} FRAUD transactions...")
    for i in range(num_fraud):
        features = create_fraud_transaction()
        result = test_single_transaction(features, "FRAUD")
        results.append(result)
        
        if "error" not in result:
            print(f"  Fraud {i+1}: Prob={result['probability']:.4f} | Pred={result['prediction']} | {'✅' if result['correct'] else '❌'}")
        else:
            print(f"  Fraud {i+1}: ❌ {result['error']}")
    
    # Summary
    print("\n📈 SUMMARY:")
    print("-" * 40)
    
    valid_results = [r for r in results if r.get("type") == "VALID" and "error" not in r]
    fraud_results = [r for r in results if r.get("type") == "FRAUD" and "error" not in r]
    
    if valid_results:
        valid_correct = sum(1 for r in valid_results if r["correct"])
        valid_avg_prob = np.mean([r["probability"] for r in valid_results])
        print(f"Valid transactions: {valid_correct}/{len(valid_results)} correct (avg prob: {valid_avg_prob:.4f})")
    
    if fraud_results:
        fraud_correct = sum(1 for r in fraud_results if r["correct"])
        fraud_avg_prob = np.mean([r["probability"] for r in fraud_results])
        print(f"Fraud transactions: {fraud_correct}/{len(fraud_results)} correct (avg prob: {fraud_avg_prob:.4f})")
    
    total_correct = sum(1 for r in results if "error" not in r and r["correct"])
    total_tests = len([r for r in results if "error" not in r])
    
    if total_tests > 0:
        accuracy = total_correct / total_tests
        print(f"Overall accuracy: {accuracy:.2%} ({total_correct}/{total_tests})")


def test_edge_cases() -> None:
    """Test edge cases and boundary conditions."""
    
    print("\n🔍 Testing Edge Cases:")
    print("-" * 30)
    
    # Test zero values
    zero_features = [0.0] * 46
    result = test_single_transaction(zero_features, "ZERO_VALUES")
    if "error" not in result:
        print(f"Zero values: Prob={result['probability']:.4f} | Pred={result['prediction']}")
    
    # Test very large values
    large_features = [1000.0] * 46
    result = test_single_transaction(large_features, "LARGE_VALUES")
    if "error" not in result:
        print(f"Large values: Prob={result['probability']:.4f} | Pred={result['prediction']}")
    
    # Test negative values
    negative_features = [-10.0] * 46
    result = test_single_transaction(negative_features, "NEGATIVE_VALUES")
    if "error" not in result:
        print(f"Negative values: Prob={result['probability']:.4f} | Pred={result['prediction']}")


if __name__ == "__main__":
    # Test multiple transactions
    test_multiple_transactions(num_valid=5, num_fraud=5)
    
    # Test edge cases
    test_edge_cases()
    
    print("\n✅ Testing completed!")
