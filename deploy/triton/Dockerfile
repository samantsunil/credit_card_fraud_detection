FROM nvcr.io/nvidia/tritonserver:24.05-py3

# Install additional Python dependencies for scikit-learn models
RUN pip install --no-cache-dir \
    joblib>=1.3.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0

# Set working directory
WORKDIR /workspace

# Expose Triton ports
EXPOSE 8000 8001 8002
