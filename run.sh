#!/usr/bin/env bash
# Starts the NewsLens web app at http://127.0.0.1:8000
cd "$(dirname "$0")"
source venv/bin/activate
echo "Open http://127.0.0.1:8000 in your browser (first start takes a few seconds to load the model)"
uvicorn backend.main:app --host 127.0.0.1 --port 8000
