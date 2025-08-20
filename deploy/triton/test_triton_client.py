#!/usr/bin/env python3
"""
Advanced test client using Triton client library
"""

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def test_with_triton_client():
    """Test using the official Triton client library."""
    
    try:
        # Create client
        client = httpclient.InferenceServerClient(url="localhost:8000")
        
        # Check if model is ready
        if not client.is_model_ready("fraud_rf_smote"):
            print("❌ Model not ready")
            return
        
        print("✅ Model is ready!")
        
        # Create test data
        test_data = np.random.randn(1, 30).astype(np.float32)
        
        # Create input tensor
        inputs = [
            httpclient.InferInput("INPUT__0", test_data.shape, np_to_triton_dtype(test_data.dtype))
        ]
        inputs[0].set_data_from_numpy(test_data)
        
        # Create output tensors
        outputs = [
            httpclient.InferRequestedOutput("OUTPUT__PROB"),
            httpclient.InferRequestedOutput("OUTPUT__CLASS")
        ]
        
        # Send inference request
        response = client.infer("fraud_rf_smote", inputs, outputs=outputs)
        
        # Get results
        prob_output = response.as_numpy("OUTPUT__PROB")
        class_output = response.as_numpy("OUTPUT__CLASS")
        
        print(f"✅ Inference successful!")
        print(f"Fraud probability: {prob_output[0][0]:.4f}")
        print(f"Prediction: {'FRAUD' if class_output[0][0] == 1 else 'LEGITIMATE'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Testing with Triton Client Library")
    print("=" * 50)
    test_with_triton_client()
