"""FastAPI service for the spoken digit classifier.

Endpoints
---------
GET  /               list the available endpoints
GET  /health         is the service (and its model) up, plus uptime
POST /predict        one audio file -> one prediction
POST /upload         many audio files -> predictions, saved to the database
GET  /uploads        the stored uploads (for the UI's data insights)
POST /retrain        warm-start from the current model on the uploaded clips
GET  /retrain/{id}   status of a retraining job
GET  /metrics        model metrics + database counts (for the UI)

Start it from the repository root:
    .venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
"""

import os
import shutil
import tempfile
import time
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.model import (
    DEFAULT_MODEL_PATH,
    METRICS_PATH,
    evaluate_model,
    load_model,
    train_model,
)
from src.preprocessing import (
    CLASS_NAMES,
    INPUT_SHAPE,
    is_supported_audio,
    load_dataset,
    prepare_batch,
    wav_to_logmel,
)
from src.prediction import get_model, predict_spectrogram, reset_model_cache

from app import db

APP_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = APP_ROOT / "data" / "uploads"
BACKUP_DIR = APP_ROOT / "models" / "backup"
ORIGINAL_TRAIN_DIR = APP_ROOT / "data" / "train"
TEST_DIR = APP_ROOT / "data" / "test"

# Overridable so tests and containers can point somewhere else.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", APP_ROOT / "data" / "app.db"))

STARTED_AT = time.time()
_test_set = None       # cached (X, y, names) of data/test for before/after accuracy

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the database tables, then load the model once (fails fast if broken)."""
    db.init(DATABASE_PATH)
    get_model(MODEL_PATH)
    yield


app = FastAPI(
    title="Spoken Digit Classifier API",
    description="Predict, store and retrain a CNN that recognises digits 0-9 in "
                "one-second audio clips.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- helpers ----
def _check_audio_name(name: str) -> str:
    """Reject anything that is not a decodable audio file name."""
    if not is_supported_audio(name):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {name} (use .wav, .flac, .ogg, .mp3)",
        )
    return name


def _save_upload(upload: UploadFile, directory: Path) -> Path:
    """Validate the extension and write the incoming bytes to disk.

    Opens the file exclusively ('xb'), so two requests that happen to send the
    same file name can never write over each other: the loser retries with a
    unique suffix instead.
    """
    name = _check_audio_name(Path(upload.filename or "clip.wav").name)
    directory.mkdir(parents=True, exist_ok=True)
    data = upload.file.read()

    target = directory / name
    for _ in range(5):
        try:
            with target.open("xb") as handle:        # exclusive create
                handle.write(data)
            return target
        except FileExistsError:
            target = directory / f"{Path(name).stem}_{uuid.uuid4().hex[:6]}{Path(name).suffix}"
    raise HTTPException(status_code=500, detail=f"could not store {name}")


def _label_from_filename(name: str) -> Optional[int]:
    """'7_george_0.wav' -> 7, anything else -> None."""
    head = Path(name).stem.split("_")[0]
    return int(head) if head.isdigit() and int(head) in range(10) else None


def _test_set_cached():
    global _test_set
    if _test_set is None:
        _test_set = load_dataset(TEST_DIR)
    return _test_set


# --------------------------------------------------------------- endpoints ---
@app.get("/")
def index():
    return {
        "service": "spoken digit classifier",
        "model": str(MODEL_PATH),
        "classes": CLASS_NAMES,
        "input_shape": INPUT_SHAPE,
        "endpoints": ["/health", "/predict", "/upload", "/uploads",
                      "/retrain", "/retrain/{job_id}", "/metrics", "/docs"],
    }


@app.get("/health")
def health():
    """Uptime check - the UI shows this."""
    return {
        "status": "ok",
        "model_loaded": get_model(MODEL_PATH) is not None,
        "model_path": str(MODEL_PATH),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Predict one uploaded audio file.

    The bytes go to a private temp file from mkstemp, so every concurrent
    request gets its own path no matter what the client called the upload.
    """
    name = _check_audio_name(Path(file.filename or "clip.wav").name)
    tmp_dir = UPLOAD_DIR / "_predict_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=tmp_dir, prefix="predict_", suffix=Path(name).suffix.lower())
    path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            shutil.copyfileobj(file.file, handle)
        result = predict_spectrogram(get_model(MODEL_PATH), wav_to_logmel(path))
    finally:
        path.unlink(missing_ok=True)

    db.add_prediction(name, result["digit"], result["confidence"],
                      "single", MODEL_PATH)
    return {"filename": name, **result}


@app.post("/upload")
def upload(
    files: List[UploadFile] = File(...),
    labels: Optional[str] = Form(None),
    use_for_training: bool = Form(True),
):
    """Bulk upload: store every file, predict it, and record it in the database.

    `labels` is optional and comma separated in the same order as the files,
    e.g. "3,3,7". Missing labels fall back to the file name ("7_george_0.wav"
    means digit 7). Files without any label can still be stored and predicted,
    they just cannot be used for training.
    """
    if not files:
        raise HTTPException(status_code=400, detail="no files were uploaded")

    given = []
    if labels:
        for chunk in labels.split(","):
            chunk = chunk.strip()
            if chunk == "":
                given.append(None)
            elif chunk.isdigit() and int(chunk) in range(10):
                given.append(int(chunk))
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"label '{chunk}' is not a digit between 0 and 9",
                )

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    batch_dir = UPLOAD_DIR / batch_id
    model = get_model(MODEL_PATH)

    saved, labelled = [], 0
    for index, upload_file in enumerate(files):
        path = _save_upload(upload_file, batch_dir)
        result = predict_spectrogram(model, wav_to_logmel(path))

        label = given[index] if index < len(given) else None
        if label is None:
            label = _label_from_filename(path.name)
        if label is not None:
            labelled += 1

        upload_id = db.add_upload(
            batch_id, path.name, path, label, result["digit"], result["confidence"])
        db.add_prediction(path.name, result["digit"], result["confidence"],
                          "bulk", MODEL_PATH)

        saved.append({
            "id": upload_id,
            "filename": path.name,
            "given_label": label,
            "predicted_label": result["digit"],
            "confidence": round(result["confidence"], 4),
            "correct": None if label is None else label == result["digit"],
            "use_for_training": bool(use_for_training and label is not None),
        })

    return {
        "batch_id": batch_id,
        "batch_dir": str(batch_dir),
        "files_saved": len(saved),
        "labelled": labelled,
        "usable_for_training": sum(1 for item in saved if item["use_for_training"]),
        "results": saved,
    }


@app.get("/uploads")
def list_uploads(limit: int = 100):
    """Stored uploads - the UI turns these into the data insights."""
    return {"counts": db.upload_counts(), "uploads": db.list_uploads(limit)}


@app.post("/retrain")
def retrain(
    epochs: int = 25,
    include_original: bool = True,
    background: bool = True,
):
    """Retrain by warm-starting from the current model on the uploaded clips.

    include_original=True also replays data/train, so a handful of new uploads
    does not wipe out what the model already knows. Set it to False to train on
    the uploads alone.
    """
    rows = db.labelled_uploads()
    if len(rows) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"need at least 5 labelled uploads to retrain, found {len(rows)}. "
                   "Upload clips with labels first (POST /upload).",
        )

    job_id = "job-" + uuid.uuid4().hex[:8]
    db.create_job(job_id, epochs)

    if background:
        threading.Thread(target=_run_retrain,
                         args=(job_id, epochs, include_original),
                         daemon=True).start()
        return {"job_id": job_id, "status": "running",
                "poll": f"/retrain/{job_id}"}

    _run_retrain(job_id, epochs, include_original)
    return db.get_job(job_id)


def _run_retrain(job_id, epochs, include_original):
    """The actual retraining work - runs in a worker thread."""
    try:
        rows = db.labelled_uploads()
        upload_paths = [Path(row["stored_path"]) for row in rows]
        upload_labels = [int(row["source_label"]) for row in rows]

        X_new = prepare_batch(upload_paths)
        y_new = upload_labels[: len(X_new)]

        if include_original:
            X_old, y_old, _ = load_dataset(ORIGINAL_TRAIN_DIR)
            X_train = np.concatenate([X_old, X_new], axis=0)
            y_train = np.concatenate([y_old, np.array(y_new)], axis=0)
        else:
            X_train, y_train = X_new, y_new

        # accuracy of the model we are about to improve, on the fixed test set.
        # Load it from disk instead of reusing the cached model: this worker
        # thread then owns every model object it touches.
        X_test, y_test, _ = _test_set_cached()
        previous_model = load_model(MODEL_PATH)
        before_metrics = evaluate_model(previous_model, X_test, y_test)
        accuracy_before = before_metrics["accuracy"]
        del previous_model

        # keep the previous weights so retraining can never destroy them silently
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BACKUP_DIR / f"spoken_digit_{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.keras"
        if MODEL_PATH.exists():
            shutil.copy2(MODEL_PATH, backup_path)
        else:
            backup_path = None

        model, history = train_model(
            X_train, y_train,
            epochs=epochs,
            learning_rate=3e-4,                 # fine-tuning, not starting from scratch
            model_path=MODEL_PATH,
            initial_model_path=backup_path,     # <-- warm start from our own model
            verbose=0,
        )
        reset_model_cache()

        after = evaluate_model(model, X_test, y_test)

        # Never let a retraining run silently make production worse. A drop
        # smaller than TOLERANCE is treated as noise: on 600 test clips, half a
        # point is only three clips.
        TOLERANCE = 0.005
        if after["accuracy"] < accuracy_before - TOLERANCE and backup_path is not None:
            shutil.copy2(backup_path, MODEL_PATH)
            reset_model_cache()
            serving_accuracy = accuracy_before
            serving_f1 = before_metrics["f1_macro"]
            decision = (f"kept the previous model: the retrained one scored "
                        f"{after['accuracy']:.4f} vs {accuracy_before:.4f}")
        else:
            serving_accuracy = after["accuracy"]
            serving_f1 = after["f1_macro"]
            decision = (f"saved the retrained model "
                        f"({before_metrics['accuracy']:.4f} -> {after['accuracy']:.4f})")

        db.mark_uploads_used([row["id"] for row in rows])
        db.update_job(
            job_id,
            status="finished",
            samples=int(len(y_train)),
            accuracy_before=accuracy_before,
            accuracy_after=serving_accuracy,
            f1_after=serving_f1,
            model_path=str(MODEL_PATH),
            message=(f"trained on {len(y_train)} clips "
                     f"({len(X_new)} uploaded + {len(y_train) - len(X_new)} original), "
                     f"{len(history.history['loss'])} epochs; {decision}; "
                     f"backup: {backup_path.name if backup_path else 'none'}"),
            finished_at=db.now(),
        )
    except Exception as error:                       # keep the job row honest
        traceback.print_exc()                        # full stack in the server log
        db.update_job(
            job_id,
            status="failed",
            message=f"{type(error).__name__}: {error}".split(chr(10))[0][:500],
            finished_at=db.now(),
        )


@app.get("/retrain/{job_id}")
def retrain_status(job_id: str):
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}")
    return job


@app.get("/retrain")
def retrain_history(limit: int = 10):
    return {"jobs": db.list_jobs(limit)}


@app.get("/metrics")
def metrics():
    """Model metrics from training plus live database counts (the UI reads this)."""
    import json

    trained = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    return {
        "trained_metrics": trained,
        "current_model": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "classes": CLASS_NAMES,
        "uploads": db.upload_counts(),
        "predictions_per_class": db.prediction_counts(),
        "recent_predictions": db.recent_predictions(20),
        "latest_retrain": db.latest_job(),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
    }
