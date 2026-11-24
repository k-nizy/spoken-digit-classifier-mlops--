"""Prediction: load the trained model once, then label uploaded clips."""

import numpy as np

from src.preprocessing import CLASS_NAMES, wav_to_logmel
from src.model import DEFAULT_MODEL_PATH, load_model

# One model in memory per process; the API reuses it across requests.
_cached_model = None
_cached_path = None


def get_model(model_path=DEFAULT_MODEL_PATH):
    """Return the trained model, loading it from disk on first use."""
    global _cached_model, _cached_path
    if _cached_model is None or _cached_path != str(model_path):
        _cached_model = load_model(model_path)
        _cached_path = str(model_path)
    return _cached_model


def reset_model_cache():
    """Forget the cached model so the next call reloads it (after retraining)."""
    global _cached_model, _cached_path
    _cached_model = None
    _cached_path = None


def predict_spectrogram(model, spectrogram):
    """Classify one log-mel spectrogram and return a readable result dict."""
    batch = np.expand_dims(spectrogram, 0)
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return {
        "digit": index,
        "label": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {
            name: float(p) for name, p in zip(CLASS_NAMES, probabilities)
        },
    }


def predict_file(path, model_path=DEFAULT_MODEL_PATH):
    """Audio file -> prediction dict (digit, label, confidence, probabilities)."""
    model = get_model(model_path)
    result = predict_spectrogram(model, wav_to_logmel(path))
    result["filename"] = str(path)
    return result


def predict_files(paths, model_path=DEFAULT_MODEL_PATH):
    """Predict a list of audio files in one batch."""
    return [predict_file(p, model_path) for p in paths]
