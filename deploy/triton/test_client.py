#!/usr/bin/env python3
"""
Test client for Triton Inference Server - Credit Card Fraud Detection
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


def create_test_data(num_samples: int = 5) -> List[Dict[str, Any]]:
    """Create realistic test data for credit card fraud detection."""
    
    # Sample feature values based on typical credit card transaction patterns
    # Features: Time, V1-V28, Amount
    test_samples = []
    
    for i in range(num_samples):
        # Create realistic feature values
        features = []
        
        # Time (seconds from first transaction)
        features.append(float(i * 1000))
        
        # V1-V28 (PCA components - typically small values)
        for j in range(28):
            features.append(np.random.normal(0, 1))
        
        # Amount (transaction amount - typically positive)
        features.append(abs(np.random.normal(100, 50)))
        
        # Add enhanced features
        enhanced_features = calculate_enhanced_features(features)
        features.extend(enhanced_features)
        
        test_samples.append(features)
    
    return test_samples


def test_triton_model(model_name: str = "fraud_rf_smote", 
                     url: str = "http://localhost:8000",
                     num_samples: int = 3) -> None:
    """Test the deployed Triton model."""
    
    # Create test data
    test_features = create_test_data(num_samples)
    
    # Prepare request with single input containing all samples
    request_data = {
        "inputs": [
            {
                "name": "INPUT__0",
                "datatype": "FP32",
                "shape": [num_samples, 46],
                "data": [feature for sample in test_features for feature in sample]
            }
        ],
        "outputs": [
            {"name": "OUTPUT__PROB"},
            {"name": "OUTPUT__CLASS"}
        ]
    }
    
    # Send request
    try:
        response = requests.post(
            f"{url}/v2/models/{model_name}/infer",
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model inference successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Parse and display results
            print("\n📊 Inference Results:")
            outputs = result.get("outputs", [])
            
            # Find probability and class outputs
            prob_output = None
            class_output = None
            for output in outputs:
                if output["name"] == "OUTPUT__PROB":
                    prob_output = output["data"]
                elif output["name"] == "OUTPUT__CLASS":
                    class_output = output["data"]
            
            # Display results for each sample
            for i in range(num_samples):
                prob = prob_output[i] if prob_output else 0.0
                pred_class = class_output[i] if class_output else 0
                fraud_status = "FRAUD" if pred_class == 1 else "LEGITIMATE"
                print(f"Sample {i+1} - Fraud Probability: {prob:.4f} | Prediction: {fraud_status}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure Triton server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_model_metadata(url: str = "http://localhost:8000", 
                       model_name: str = "fraud_rf_smote") -> None:
    """Test model metadata endpoint."""
    
    try:
        response = requests.get(f"{url}/v2/models/{model_name}")
        if response.status_code == 200:
            metadata = response.json()
            print("✅ Model metadata retrieved successfully!")
            print(f"Model: {metadata.get('name')}")
            print(f"Platform: {metadata.get('platform')}")
            print(f"Versions: {metadata.get('versions')}")
        else:
            print(f"❌ Error getting metadata: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Testing Triton Inference Server - Credit Card Fraud Detection")
    print("=" * 70)
    
    # Test model metadata
    print("\n1. Testing model metadata...")
    test_model_metadata()
    
    # Test model inference
    print("\n2. Testing model inference...")
    test_triton_model()
    
    print("\n✅ Testing completed!")
