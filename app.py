import streamlit as st
import numpy as np
import pickle, io, os, requests
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import time

st.set_page_config(layout="wide", page_title="Wearable ECG Streaming Demo")

# ==============================
# 1) GOOGLE DRIVE DIRECT LINKS
# ==============================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1iEWlaP-XB30Rxwyz8wIYcfvO_hW5PR7a"
DATA_URL  = "https://drive.google.com/uc?export=download&id=1AdLKeXQI8pz2cyaK33gwpPcwJYvrebVL"

MODEL_PATH = "afdb_ecg_cnn.h5"
DATA_PATH  = "test_data.pkl"

# ==============================
# 2) DOWNLOAD IF NOT PRESENT
# ==============================
def download_if_missing(url, path):
    if not os.path.exists(path):
        st.info(f"Downloading {path}...")
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
        st.success(f"{path} downloaded.")

download_if_missing(MODEL_URL, MODEL_PATH)
download_if_missing(DATA_URL,  DATA_PATH)

# ==============================
# 3) LOAD MODEL AND DATA
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

with open(DATA_PATH, "rb") as f:
    d = pickle.load(f)

X_all = d["X_test"]
y_all = d["y_test"]
fs = d["fs"]
N = len(X_all)

# ==============================
# UI LAYOUT
# ==============================
st.title("Smart Wearable ECG Monitoring System (1D-CNN)")

col1, col2 = st.columns([1,2])

# --- Sidebar ---
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.5)
play = st.sidebar.button("Start Streaming")
pause = st.sidebar.button("Pause Streaming")
reset = st.sidebar.button("Reset Stats")

# Session state
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "cm" not in st.session_state:
    st.session_state.cm = np.zeros((2,2), dtype=int)

if play:  st.session_state.running = True
if pause: st.session_state.running = False
if reset:
    st.session_state.cm = np.zeros((2,2), dtype=int)
    st.session_state.idx = 0

# ==============================
# Helper: ECG Plot
# ==============================
def plot_ecg(sig, fs, y_true, y_pred, prob):
    fig, ax = plt.subplots(figsize=(8,2))
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, linewidth=0.8)
    ax.set_title(f"True: {y_true} | Pred: {y_pred} | Prob: {prob:.3f}")
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return fig

# ==============================
# Main streaming display
# ==============================
with col2:
    st.header("Live ECG & Predictions")
    plot_area = st.empty()
    pred_text = st.empty()

with col1:
    st.header("Metrics")
    cm_box = st.empty()
    met_box = st.empty()

# ==============================
# STREAMING LOGIC
# ==============================
if st.session_state.running:
    idx = st.session_state.idx % N

    x = X_all[idx].squeeze()
    y_true = int(y_all[idx])

    prob = float(model.predict(X_all[idx:idx+1]).ravel()[0])
    y_pred = 1 if prob >= threshold else 0

    # update confusion matrix
    st.session_state.cm[y_true, y_pred] += 1
    st.session_state.idx += 1

    # render
    fig = plot_ecg(x, fs, y_true, y_pred, prob)
    plot_area.pyplot(fig)
    pred_text.markdown(f"### Pred: **{'AF' if y_pred else 'Normal'}** — Prob = **{prob:.3f}**")

    cm = st.session_state.cm
    cm_box.write(cm)

    TP = cm[1,1]; FP = cm[0,1]; FN = cm[1,0]; TN = cm[0,0]
    precision = TP / (TP+FP+1e-8)
    recall    = TP / (TP+FN+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)

    met_box.write(f"Precision: {precision:.3f} — Recall: {recall:.3f} — F1: {f1:.3f}")

    time.sleep(0.7)
    st.experimental_rerun()

else:
    idx = st.session_state.idx % N
    x = X_all[idx].squeeze()
    y_true = int(y_all[idx])
    prob = float(model.predict(X_all[idx:idx+1]).ravel()[0])
    y_pred = 1 if prob >= threshold else 0

    fig = plot_ecg(x, fs, y_true, y_pred, prob)
    plot_area.pyplot(fig)
    pred_text.markdown(f"Pred: **{'AF' if y_pred else 'Normal'}** — Prob = **{prob:.3f}**")

    cm = st.session_state.cm
    cm_box.write(cm)
    met_box.write("Click **Start Streaming** to begin.")
