from __future__ import annotations

from pathlib import Path


ZENODO_MODEL_URL = "https://zenodo.org/records/13913710/files/Autoencoder_Meander_Bend.h5?download=1"
ZENODO_MODEL_FILENAME = "Autoencoder_Meander_Bend.h5"
ZENODO_MODEL_MD5 = "8d7fa1fe2719b052abff6f8d884c77d1"


def load_autoencoder(model_path: str | Path):
    """Load the Keras autoencoder model lazily."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load the autoencoder. Install with: "
            "python -m pip install -e '.[deep-learning]'"
        ) from exc
    return tf.keras.models.load_model(model_path)


def build_encoder_from_autoencoder(autoencoder, *, latent_dim: int = 2):
    """Build an encoder model by locating the latent layer.

    The Zenodo model uses a two-dimensional bottleneck. This helper searches for
    the last layer whose output has shape ``(..., latent_dim)`` and returns the
    model up to that layer.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow/Keras is required for encoder extraction.") from exc

    latent_layer = None
    for layer in autoencoder.layers:
        shape = getattr(layer, "output_shape", None)
        if shape is None:
            try:
                shape = layer.output.shape
            except Exception:
                shape = None
        if shape is not None and len(shape) >= 2 and shape[-1] == latent_dim:
            latent_layer = layer
    if latent_layer is None:
        raise ValueError(f"Could not locate a latent layer with dimension {latent_dim}.")
    return keras.Model(autoencoder.input, latent_layer.output)


def prepare_images_for_keras(images):
    """Return images shaped as (n, height, width, 1)."""
    import numpy as np

    arr = np.asarray(images, dtype="float32")
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError("Expected images with shape (n, h, w) or (n, h, w, 1).")
    return arr
