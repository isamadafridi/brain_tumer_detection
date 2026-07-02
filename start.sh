#!/bin/bash
# Starts FastAPI in the background, then Streamlit in the foreground.
# Both run in the same container, which is what a single HF Space needs.

set -e

uvicorn api:app --host 0.0.0.0 --port 8000 &

# Give FastAPI a moment to load the model before Streamlit starts sending requests
sleep 5

streamlit run app.py --server.port 7860 --server.address 0.0.0.0
