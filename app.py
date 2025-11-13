# app.py — Streamlit demo (Drive downloads via gdown + sanity checks)
import streamlit as st
import numpy as np
import pickle, os, time
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide", page_title="Wearable ECG Streaming Demo (gdown)")

# ================
# Google Drive IDs (keep these as-is)
# ================
MODEL_DRIVE_ID = "12daI3A_Ff8819mpAZl-pm4nfGgSC9r00"
DATA_DRIVE_ID  = "1AdLKeXQI8pz2cyaK33gwpPcwJYvrebVL"
MODEL_PATH = "afdb_ecg_cnn.h5"
DATA_PATH  = "test_data.pkl"

# ================
# Download helper using gdown (more reliable for Drive)
# ================
def download_from_drive(drive_id, out_path):
    """
    Download a Google Drive file via gdown. Returns True if file exists and is plausible.
    """
    try:
        import gdown
    except Exception:
        st.info("Installing gdown...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={drive_id}"
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        st.info(f"Downloading {out_path} from Drive...")
        gdown.download(url, out_path, quiet=False)
    # sanity check
    if not os.path.exists(out_path):
        st.error(f"Download failed — {out_path} not found.")
        return False
    size = os.path.getsize(out_path)
    if size < 1024:
        st.error(f"Downloaded {out_path} is too small ({size} bytes).")
        return False
    return True

# Try downloads
ok1 = download_from_drive(MODEL_DRIVE_ID, MODEL_PATH)
ok2 = download_from_drive(DATA_DRIVE_ID, DATA_PATH)

if not ok1 or not ok2:
    st.stop()

# Load model and data with try/except and user-visible errors
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}: {e}")
    # show file info to help debugging
    try:
        st.write("File info:", os.path.abspath(MODEL_PATH), "size=", os.path.getsize(MODEL_PATH))
        with open(MODEL_PATH, "rb") as fh:
            head = fh.read(512)
        st.write("First bytes (hex):", head[:64].hex())
    except Exception:
        pass
    st.stop()

try:
    with open(DATA_PATH, "rb") as f:
        d = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load test data from {DATA_PATH}: {e}")
    try:
        st.write("File info:", os.path.abspath(DATA_PATH), "size=", os.path.getsize(DATA_PATH))
    except Exception:
        pass
    st.stop()

X_all = d.get("X_test")
y_all = d.get("y_test")
fs = int(d.get("fs", 250))
if X_all is None or y_all is None:
    st.error("test_data.pkl does not contain X_test / y_test keys.")
    st.stop()

N = len(X_all)

# ================
# UI Layout
# ================
st.title("Smart Wearable ECG Monitoring System (1D-CNN)")

col1, col2 = st.columns([1,2])

st.sidebar.header("Controls")
threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.5)
play = st.sidebar.button("Start Streaming")
pause = st.sidebar.button("Pause Streaming")
reset = st.sidebar.button("Reset Stats")

# session state
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "running" not in st.session_state:
    st.session_state.running = False
if "cm" not in st.session_state:
    st.session_state.cm = np.zeros((2,2), dtype=int)

if play:
    st.session_state.running = True
if pause:
    st.session_state.running = False
if reset:
    st.session_state.cm = np.zeros((2,2), dtype=int)
    st.session_state.idx = 0

# ECG plotting helper
def plot_ecg(sig, fs, y_true, y_pred, prob):
    fig, ax = plt.subplots(figsize=(8,2))
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, linewidth=0.8)
    ax.set_title(f"True: {y_true} | Pred: {y_pred} | Prob: {prob:.3f}")
    ax.set_ylim(-3, 3)
    ax.set_xlabel("Time (s)")
    plt.tight_layout()
    return fig

with col2:
    st.header("Live ECG & Predictions")
    plot_area = st.empty()
    pred_text = st.empty()

with col1:
    st.header("Metrics")
    cm_box = st.empty()
    met_box = st.empty()

# streaming loop
if st.session_state.running:
    idx = st.session_state.idx % N
    x = X_all[idx].squeeze()
    y_true = int(y_all[idx])
    prob = float(model.predict(X_all[idx:idx+1]).ravel()[0])
    y_pred = 1 if prob >= threshold else 0
    st.session_state.cm[y_true, y_pred] += 1
    st.session_state.idx += 1
    fig = plot_ecg(x, fs, y_true, y_pred, prob)
    plot_area.pyplot(fig)
    pred_text.markdown(f"### Pred: **{'AF' if y_pred else 'Normal'}**  Prob: **{prob:.3f}**")
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
    pred_text.markdown(f"Pred: **{'AF' if y_pred else 'Normal'}**  Prob: **{prob:.3f}**")
    cm_box.write(st.session_state.cm)
    met_box.write("Click Start Streaming to begin.")

