#!/bin/bash
set -e

echo "== Installing system dependencies =="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1

echo "== Installing Python packages =="
pip install -q -r req.txt

echo "== Training the model =="
python train_model.py

echo ""
echo "Setup complete! Run:  streamlit run app.py"