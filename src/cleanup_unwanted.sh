#!/usr/bin/env bash
set -euo pipefail

echo "Cleaning up unwanted Triton client files in deploy/triton..."
rm -vf deploy/triton/test_client.py deploy/triton/test_triton_client.py || true

# remove any stale pyc/__pycache__ for those files
find deploy/triton -type d -name "__pycache__" -print0 | xargs -0 -r rm -rvf || true

echo "Cleanup complete."