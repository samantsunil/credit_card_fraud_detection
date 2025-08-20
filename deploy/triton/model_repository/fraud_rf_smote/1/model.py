import os
import numpy as np
import joblib
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # Handle different argument formats
        if isinstance(args, dict):
            model_config = args.get("model_config", {})
        else:
            model_config = {}
            
        # Get parameters safely
        params = {}
        if isinstance(model_config, dict):
            params = model_config.get("parameters", {})
        
        # Get model filename
        model_filename = "random_forest_smote.joblib"
        if "MODEL_FILENAME" in params:
            model_filename_param = params["MODEL_FILENAME"]
            if isinstance(model_filename_param, dict) and "string_value" in model_filename_param:
                model_filename = model_filename_param["string_value"]

        # Load the RF model artifact from the same version directory
        model_dir = os.path.dirname(__file__)
        artifact_path = os.path.join(model_dir, model_filename)
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        self.model = joblib.load(artifact_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
            data = in_tensor.as_numpy()  # shape: [B, 30]

            # Ensure 2D
            if data.ndim == 1:
                data = np.expand_dims(data, 0)

            # Predict probabilities if available; fallback to decision function
            try:
                proba = self.model.predict_proba(data)[:, 1].astype(np.float32)
            except Exception:
                # Fallback: convert decision function/logits to pseudo-prob via sigmoid
                logits = self.model.decision_function(data).astype(np.float32)
                proba = 1.0 / (1.0 + np.exp(-logits))

            preds = (proba >= 0.5).astype(np.int64)

            out_prob = pb_utils.Tensor("OUTPUT__PROB", proba.reshape(-1, 1))
            out_class = pb_utils.Tensor("OUTPUT__CLASS", preds.reshape(-1, 1))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_prob, out_class])
            responses.append(inference_response)

        return responses

    def finalize(self):
        # Nothing to clean up
        pass


