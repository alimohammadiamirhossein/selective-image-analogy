#!/bin/bash
# Install the package and download the SIA training data into ./data
# (from the Hugging Face Hub: Amirhossein-Alimohammadi/SIA-Multi-Edit-Analogy).
set -e

pip install -r requirements.txt
pip install -e .

python scripts/download_data.py --out data
