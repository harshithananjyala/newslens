#!/usr/bin/env bash
# One-time setup for macOS: creates a virtual environment and installs dependencies.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "Using $($PY --version)"
$PY -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo ""
echo "Setup complete. Next step:  source venv/bin/activate && python -m backend.ingest"
