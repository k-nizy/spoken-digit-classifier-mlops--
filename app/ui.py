"""Streamlit dashboard for the spoken digit classifier.

Run it from the repository root, with the API already running:

    .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
    .venv/Scripts/python.exe -m streamlit run app/ui.py

What it shows
-------------
1. Overview        - is the API up, how long has it been up, model metrics
2. Feature insights- the three charts that explain the data
3. Predict         - one clip -> digit + confidence + probability bars
4. Upload & retrain- bulk upload to the database, then retrain from it

The API address comes from the API_URL environment variable
(default http://127.0.0.1:8000).
"""

import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                     # so "src" imports work here
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (                   # noqa: E402
    CLASS_NAMES,
    HOP_LENGTH,
    N_MELS,
    SR,
    load_audio,
    logmel_from_array,
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
TEST_DIR = ROOT / "data" / "test"
CLIPS_PER_DIGIT = 20                              # enough for a stable average


# ------------------------------------------------------------------- API -----
def api_get(path, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path, **kwargs):
    response = requests.post(f"{API_URL}{path}", timeout=300, **kwargs)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def human_uptime(seconds):
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = [f"{days}d", f"{hours}h", f"{minutes}m", f"{seconds}s"]
    return " ".join(part for part in parts[: 1 if days else 2 if hours else 3])


# --------------------------------------------------------------- features ----
@st.cache_data(show_spinner="Reading a sample of test clips ...")
def feature_tables(clips_per_digit=CLIPS_PER_DIGIT):
    """Build the numbers behind the three charts from data/test.

    Returns
    -------
    mean_specs : dict digit -> (N_MELS, frames) averaged log-mel spectrogram
    per_digit  : dict digit -> {'centroid': [...], 'rms': [...], 'duration': [...]}
    """
    mean_specs, per_digit = {}, {}
    for digit in CLASS_NAMES:
        folder = TEST_DIR / digit
        if not folder.is_dir():
            continue
        specs, centroids, rms, durations = [], [], [], []
        for clip in sorted(folder.glob("*.wav"))[:clips_per_digit]:
            y = load_audio(clip)
            specs.append(logmel_from_array(y))
            centroid = librosa_centroid(y)
            centroids.append(centroid)
            rms.append(float(np.sqrt(np.mean(y ** 2))))
            durations.append(len(y) / SR)
        if not specs:
            continue
        mean_specs[int(digit)] = np.mean(np.stack(specs), axis=0)
        per_digit[int(digit)] = {
            "centroid": centroids, "rms": rms, "duration": durations,
        }
    return mean_specs, per_digit


def librosa_centroid(y):
    """Average spectral centroid (Hz) of one clip - a 'brightness' measure."""
    import librosa

    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=512,
                                                 hop_length=HOP_LENGTH)
    return float(np.mean(centroid))


def plot_mean_spectrograms(mean_specs):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.2), sharey=True)
    vmin = min(spec.min() for spec in mean_specs.values())
    vmax = max(spec.max() for spec in mean_specs.values())
    for axis, (digit, spec) in zip(axes.ravel(), sorted(mean_specs.items())):
        image = axis.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                            vmin=vmin, vmax=vmax)
        axis.set_title(f"digit {digit}", fontsize=10)
        axis.set_xlabel("frames", fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in axes[:, 0]:
        axis.set_ylabel("mel band", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=0.7, label="scaled dB")
    return fig


def plot_centroid(per_digit):
    fig, axis = plt.subplots(figsize=(9, 4))
    digits = sorted(per_digit)
    axis.boxplot([per_digit[d]["centroid"] for d in digits], tick_labels=digits)
    axis.set_xlabel("digit")
    axis.set_ylabel("mean spectral centroid (Hz)")
    axis.set_title("Spectral centroid per digit")
    axis.grid(alpha=0.3, axis="y")
    return fig


def plot_energy_vs_duration(per_digit):
    fig, axis = plt.subplots(figsize=(9, 4.2))
    for digit in sorted(per_digit):
        axis.scatter(per_digit[digit]["duration"], per_digit[digit]["rms"],
                     s=22, alpha=0.75, label=str(digit))
    axis.set_xlabel("clip length (s, after padding)")
    axis.set_ylabel("RMS energy")
    axis.set_title("Energy vs length per digit")
    axis.legend(title="digit", ncol=5, fontsize=8)
    axis.grid(alpha=0.3)
    return fig


# ------------------------------------------------------------------- pages ---
def overview_tab(health, metrics):
    left, middle, right, fourth = st.columns(4)
    left.metric("Service", health["status"].upper())
    middle.metric("Uptime", human_uptime(health["uptime_seconds"]))
    right.metric("Model loaded", "yes" if health["model_loaded"] else "NO")
    fourth.metric("Uploads stored", metrics["uploads"]["total"])

    st.caption(f"API: {API_URL}  ·  model: {health['model_path']}  ·  "
               f"checked {health['checked_at']}")

    trained = metrics.get("trained_metrics", {})
    if trained:
        st.subheader("Metrics of the trained model")
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{trained.get('accuracy', 0):.4f}")
        cols[1].metric("Precision (macro)", f"{trained.get('precision_macro', 0):.4f}")
        cols[2].metric("Recall (macro)", f"{trained.get('recall_macro', 0):.4f}")
        cols[3].metric("F1 (macro)", f"{trained.get('f1_macro', 0):.4f}")

    latest = metrics.get("latest_retrain")
    if latest:
        st.subheader("Latest retraining job")
        st.write(f"`{latest['id']}` · **{latest['status']}** · "
                 f"accuracy {latest.get('accuracy_before')} → {latest.get('accuracy_after')}")
        st.caption(latest.get("message") or "")

    st.subheader("Database counters")
    first, second = st.columns(2)
    first.write("**Uploads per batch**")
    first.json(metrics["uploads"])
    second.write("**Predictions per class**")
    second.json(metrics["predictions_per_class"])


def insights_tab():
    st.subheader("1. Average spectrogram per digit")
    st.pyplot(plot_mean_spectrograms(feature_tables()[0]))
    st.caption("Bright means energy at that frequency: every digit has a strong low "
               "band at the bottom (the vowel), and the digits with a hiss in them - "
               "six, nine, seven - also light up the top, which is why six has the "
               "brightest spectrum (centroid about 980 Hz) and four the darkest "
               "(about 430 Hz).")

    st.subheader("2. Spectral centroid per digit")
    st.pyplot(plot_centroid(feature_tables()[1]))
    st.caption("The taller the box, the brighter the digit sounds: six is highest at "
               "about 980 Hz because of the hiss at the start of 'six', four is lowest "
               "at about 430 Hz because it is mostly one vowel - and the boxes overlap "
               "heavily, so brightness alone cannot separate the digits; the model needs "
               "the whole spectrogram.")

    st.subheader("3. Energy vs clip length per digit")
    st.pyplot(plot_energy_vs_duration(feature_tables()[1]))
    st.caption("Every clip is padded or trimmed to exactly 1.0 s, so all the points "
               "sit on one vertical line and the x-axis adds nothing - only loudness "
               "varies, by about 1.7x between the quietest and the loudest digit, so a "
               "loud 'two' and a quiet 'two' look like different images to the model.")


def predict_tab():
    with st.form("predict_form"):
        clip = st.file_uploader("One audio clip", type=["wav", "flac", "ogg", "mp3"])
        submitted = st.form_submit_button("Predict")
    if submitted:
        if clip is None:
            st.warning("Choose a file first.")
        else:
            with st.spinner("Sending the clip to the API ..."):
                try:
                    result = api_post("/predict",
                                      files={"file": (clip.name, clip.getvalue())})
                except (RuntimeError, requests.RequestException) as error:
                    st.error(f"Prediction failed: {error}")
                    return
            st.metric("Predicted digit", result["digit"])
            st.metric("Confidence", f"{result['confidence']:.4f}")
            probabilities = result.get("probabilities") or {}
            if probabilities:
                fig, axis = plt.subplots(figsize=(8, 3))
                axis.bar([str(k) for k in probabilities],
                         list(probabilities.values()), color="#4C78A8")
                axis.set_xlabel("digit")
                axis.set_ylabel("probability")
                axis.set_ylim(0, 1)
                st.pyplot(fig)


def upload_retrain_tab(metrics):
    st.subheader("Bulk upload")
    with st.form("upload_form"):
        files = st.file_uploader(
            "Audio clips", type=["wav", "flac", "ogg", "mp3"], accept_multiple_files=True)
        labels = st.text_input(
            "Labels (optional, comma separated in the same order as the files)",
            placeholder="3,3,7")
        use_for_training = st.checkbox(
            "Use these clips for the next retraining run", value=True)
        uploaded = st.form_submit_button("Upload and predict")

    if uploaded:
        if not files:
            st.warning("Choose at least one file.")
        else:
            payload = [("files", (f.name, f.getvalue())) for f in files]
            data = {"use_for_training": str(use_for_training).lower()}
            if labels.strip():
                data["labels"] = labels.strip()
            with st.spinner(f"Uploading {len(files)} clip(s) ..."):
                try:
                    result = api_post("/upload", files=payload, data=data)
                except (RuntimeError, requests.RequestException) as error:
                    st.error(f"Upload failed: {error}")
                    return
            st.success(f"Batch `{result['batch_id']}`: {result['files_saved']} file(s), "
                       f"{result['labelled']} labelled, "
                       f"{result['usable_for_training']} usable for training.")
            st.dataframe(result["results"], use_container_width=True)

    st.subheader("Retrain")
    stored = metrics["uploads"]
    st.write(f"The database holds **{stored['total']}** uploads, "
             f"**{stored['labelled']}** of them labelled "
             f"(at least 5 labelled clips are needed).")

    with st.form("retrain_form"):
        epochs = st.slider("Fine-tuning epochs", 5, 40, 15)
        include_original = st.checkbox(
            "Also replay data/train (keeps what the model already knows)", value=True)
        started = st.form_submit_button("Start retraining")

    if started:
        try:
            job = api_post("/retrain", params={"epochs": epochs,
                                               "include_original": include_original,
                                               "background": True})
        except (RuntimeError, requests.RequestException) as error:
            st.error(f"Could not start retraining: {error}")
            return

        job_id = job["job_id"]
        with st.status(f"Retraining job {job_id} ...", expanded=True) as status:
            for _ in range(180):                        # up to 6 minutes
                time.sleep(2)
                try:
                    job = api_get(f"/retrain/{job_id}")
                except (RuntimeError, requests.RequestException) as error:
                    status.write(f"status check failed: {error}")
                    continue
                status.write(f"{job['status']} · {job.get('message') or ''}")
                if job["status"] != "running":
                    break
            if job["status"] == "finished":
                status.update(label=f"Retraining {job_id} finished", state="complete")
            elif job["status"] == "running":
                status.update(label=f"{job_id} still running - check back later",
                              state="error")
            else:
                status.update(label=f"Retraining {job_id} failed", state="error")

        if job["status"] == "finished":
            left, middle, right = st.columns(3)
            left.metric("Accuracy before", f"{job['accuracy_before']:.4f}")
            middle.metric("Accuracy after", f"{job['accuracy_after']:.4f}",
                          delta=f"{job['accuracy_after'] - job['accuracy_before']:+.4f}")
            right.metric("Clips trained on", job["samples"])
            st.caption(job.get("message") or "")


# -------------------------------------------------------------------- main ---
def main():
    st.set_page_config(page_title="Spoken digit classifier", page_icon="🔊",
                       layout="wide")

    st.title("Spoken digit classifier")
    st.caption("CNN on log-mel spectrograms of one-second FSDD clips (digits 0-9).")

    try:
        health = api_get("/health")
        metrics = api_get("/metrics")
    except requests.RequestException as error:
        st.error(f"Cannot reach the API at {API_URL}: {error}")
        st.info("Start it first: "
                "`.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000`")
        return

    tabs = st.tabs(["Overview", "Feature insights", "Predict", "Upload & retrain"])
    with tabs[0]:
        overview_tab(health, metrics)
    with tabs[1]:
        if TEST_DIR.is_dir():
            insights_tab()
        else:
            st.warning(f"No test data found at {TEST_DIR}.")
    with tabs[2]:
        predict_tab()
    with tabs[3]:
        upload_retrain_tab(metrics)


main()
