# Wearable ECG Monitor — Streamlit Demo

This repository contains a Streamlit demo that simulates a wearable ECG monitor.
The app streams 10s ECG windows to a 1D-CNN and shows live predictions and metrics.

Files:
- `app.py` — Streamlit app (downloads model & data from Google Drive)
- `requirements.txt` — needed Python packages
- `.gitignore` — excludes large model/data files (we store those on Drive)

Notes:
- The model (`afdb_ecg_cnn.h5`) and test data (`test_data.pkl`) are downloaded at runtime from Google Drive.
- Do not upload large binary files to the repo. Instead keep them on Drive or object storage.

Deployment:
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and deploy the repo.

