"""Audio preprocessing: one clip -> one log-mel spectrogram image.

This is the exact pipeline the notebook uses, moved here so the API, the
retraining script and the tests all share one implementation.

Pipeline
--------
1. load the file as mono float32 at 8 kHz (FSDD clips are already 8 kHz)
2. zero-pad or trim to exactly DURATION seconds
3. mel-spectrogram with N_MELS bands
4. convert to decibels, then min-max scale to 0..1

The result is a (N_MELS, frames) image, so a CNN built for images can be used
on sound.
"""

from pathlib import Path

import librosa
import numpy as np

# --- pipeline settings (must match the notebook) -----------------------------
SR = 8000                      # sample rate in Hz
DURATION = 1.0                 # seconds per clip after padding/trimming
N_MELS = 64                    # mel bands -> image height
N_FFT = 512                    # window size for the STFT
HOP_LENGTH = 128               # step between windows -> image width
N_FRAMES = 1 + int(SR * DURATION) // HOP_LENGTH          # 63
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)                      # (64, 63, 1)

# --- labels ------------------------------------------------------------------
CLASS_NAMES = [str(d) for d in range(10)]                # '0' ... '9'
NUM_CLASSES = len(CLASS_NAMES)

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def load_audio(path, sr=SR, duration=DURATION):
    """Read one audio file as a fixed-length mono float32 waveform."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    return np.asarray(y[:target_length], dtype="float32")


def logmel_from_array(y, sr=SR, n_mels=N_MELS):
    """Turn a waveform into a normalised 0..1 log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    scaled = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return scaled.astype("float32")


def wav_to_logmel(path):
    """Convenience wrapper: audio file -> log-mel spectrogram."""
    return logmel_from_array(load_audio(path))


def is_supported_audio(path):
    """True if the file extension looks like audio we can decode."""
    return Path(path).suffix.lower() in AUDIO_SUFFIXES


def prepare_batch(paths):
    """Turn a list of audio files into one model-ready array (n, 64, 63, 1)."""
    if not paths:
        raise ValueError("no audio files given")
    specs = [wav_to_logmel(p) for p in paths]
    return np.expand_dims(np.stack(specs), -1)


def load_dataset(directory):
    """Read a data/<split>/<digit>/*.wav tree into (X, y, file names).

    The folder name is the label, so the directory layout carries the truth.
    """
    directory = Path(directory)
    X, y, names = [], [], []
    class_dirs = sorted(
        (d for d in directory.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    for class_dir in class_dirs:
        for audio_path in sorted(class_dir.iterdir()):
            if not is_supported_audio(audio_path):
                continue
            X.append(wav_to_logmel(audio_path))
            y.append(int(class_dir.name))
            names.append(audio_path.name)
    if not X:
        raise FileNotFoundError(f"no labelled audio found in {directory}")
    return np.expand_dims(np.stack(X), -1), np.array(y), names
