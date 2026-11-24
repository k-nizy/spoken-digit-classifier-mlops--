"""Model definition, training and evaluation for the spoken digit classifier.

`build_model` is the same architecture as the notebook. `train_model` holds the
early-stopping training loop and can start from an already-saved model, which is
what the retraining endpoint in step 3 will use.
"""

from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocessing import CLASS_NAMES, INPUT_SHAPE, NUM_CLASSES

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "spoken_digit_cnn.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"


def build_model(input_shape=INPUT_SHAPE, n_classes=NUM_CLASSES):
    """Three convolutional blocks + a dense head, as trained in the notebook."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ],
        name="spoken_digit_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model(model, path=DEFAULT_MODEL_PATH):
    """Save the full model (architecture + weights + compile state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_model(path=DEFAULT_MODEL_PATH):
    """Load a saved model from disk."""
    return keras.models.load_model(Path(path))


def train_model(
    X,
    y,
    epochs=60,
    batch_size=32,
    validation_split=0.15,
    patience=8,
    learning_rate=1e-3,
    seed=42,
    model_path=DEFAULT_MODEL_PATH,
    initial_model_path=None,
    verbose=2,
):
    """Train the CNN with early stopping and save the best model.

    Parameters
    ----------
    X, y : arrays from preprocessing.load_dataset or preprocessing.prepare_batch.
    initial_model_path : optional path to an existing model. When given, its
        weights are loaded first, so training continues from our own pretrained
        model instead of starting from random weights (this is the retraining
        path used by the API).
    model_path : where the trained model is written.

    Returns (model, history).
    """
    keras.utils.set_random_seed(seed)

    model = build_model(input_shape=X.shape[1:])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    if initial_model_path is not None:
        model.load_weights(str(initial_model_path))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
    ]

    history = model.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Make sure the file on disk is the one early stopping kept.
    save_model(model, model_path)
    return model, history


def evaluate_model(model, X, y):
    """Compute the metrics reported in the notebook, as a plain dict."""
    loss, _ = model.evaluate(X, y, verbose=0)
    y_pred = model.predict(X, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y, y_pred, labels=range(NUM_CLASSES))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "test_clips": int(len(y)),
        "per_class_accuracy": {
            name: float(value) for name, value in zip(CLASS_NAMES, per_class)
        },
        "report": classification_report(
            y, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }
