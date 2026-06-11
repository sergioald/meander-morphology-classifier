from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


ZENODO_MODEL_URL = "https://zenodo.org/records/13913710/files/Autoencoder_Meander_Bend.h5?download=1"
ZENODO_MODEL_FILENAME = "Autoencoder_Meander_Bend.h5"
ZENODO_MODEL_MD5 = "8d7fa1fe2719b052abff6f8d884c77d1"


def _remove_default_groups_from_config(config: Any) -> Any:
    """Remove legacy/forward-incompatible ``groups=1`` entries from Keras configs.

    Some TensorFlow/Keras combinations save ``groups: 1`` in convolution layer
    configs, while other combinations reject that argument when loading an H5
    model. Removing only the default value preserves the layer behaviour.
    """
    if isinstance(config, dict):
        cleaned = {}
        for key, value in config.items():
            if key == "groups" and value == 1:
                continue
            cleaned[key] = _remove_default_groups_from_config(value)
        return cleaned
    if isinstance(config, list):
        return [_remove_default_groups_from_config(item) for item in config]
    return config


def _write_sanitized_h5_copy(model_path: str | Path) -> Path:
    """Create a temporary H5 copy with sanitized Keras model_config metadata."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to repair legacy Keras H5 model metadata.") from exc

    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(source)

    tmp = tempfile.NamedTemporaryFile(prefix="meander_legacy_model_", suffix=".h5", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    shutil.copy2(source, tmp_path)

    with h5py.File(tmp_path, "r+") as handle:
        raw_config = handle.attrs.get("model_config")
        if raw_config is None:
            return tmp_path
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")
        config = json.loads(raw_config)
        cleaned = _remove_default_groups_from_config(config)
        handle.attrs.modify("model_config", json.dumps(cleaned))

    return tmp_path


def _legacy_custom_objects(tf):
    """Return custom objects for loading legacy Zenodo/Keras H5 models."""

    class CompatibleConv2DTranspose(tf.keras.layers.Conv2DTranspose):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

        @classmethod
        def from_config(cls, config):
            config = dict(config)
            config.pop("groups", None)
            return cls(**config)

    return {
        "Conv2DTranspose": CompatibleConv2DTranspose,
        "leaky_relu": tf.nn.leaky_relu,
    }


def load_autoencoder(model_path: str | Path):
    """Load the Keras autoencoder model lazily.

    The Zenodo H5 model may contain ``groups=1`` in ``Conv2DTranspose`` layer
    metadata. That is harmless, but some TensorFlow/Keras versions reject it.
    This loader first tries the normal path, then falls back to a sanitized
    temporary H5 copy with the default ``groups`` metadata removed.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load the autoencoder. Install with: "
            "python -m pip install -e '.[deep-learning]'"
        ) from exc

    model_path = Path(model_path)
    custom_objects = _legacy_custom_objects(tf)

    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
        )
    except Exception as first_error:
        message = str(first_error)
        should_retry = "groups" in message or "Conv2DTranspose" in message
        if not should_retry:
            raise

        repaired_path: Path | None = None
        try:
            repaired_path = _write_sanitized_h5_copy(model_path)
            return tf.keras.models.load_model(
                repaired_path,
                compile=False,
                custom_objects=custom_objects,
            )
        except Exception as second_error:
            raise RuntimeError(
                "Could not load the Zenodo autoencoder with this TensorFlow/Keras environment. "
                "The model file is a legacy H5 file; try TensorFlow 2.12--2.15, or update the "
                "environment and rerun the GUI. Original loading error: "
                f"{first_error}. Fallback loading error: {second_error}"
            ) from second_error
        finally:
            if repaired_path is not None:
                try:
                    repaired_path.unlink(missing_ok=True)
                except Exception:
                    pass


def _shape_tuple(shape) -> tuple[int | None, ...] | None:
    if shape is None:
        return None
    try:
        return tuple(None if dim is None else int(dim) for dim in shape)
    except Exception:
        return None


def _get_layer_output_shape(layer):
    shape = getattr(layer, "output_shape", None)
    if shape is None:
        try:
            shape = layer.output.shape
        except Exception:
            shape = None
    return _shape_tuple(shape)


def _get_layer_input_shape(layer):
    shape = getattr(layer, "input_shape", None)
    if shape is None:
        try:
            shape = layer.input.shape
        except Exception:
            shape = None
    return _shape_tuple(shape)


def build_encoder_from_autoencoder(autoencoder, *, latent_dim: int = 2):
    """Build an encoder model by locating the latent bottleneck.

    The Zenodo model uses a two-dimensional bottleneck. This helper first looks
    for a layer whose output is ``(batch, latent_dim)``. If that fails, it uses
    the legacy-script strategy: find the first decoder layer whose input is
    ``(batch, latent_dim)`` and return the model up to the previous layer.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow/Keras is required for encoder extraction.") from exc

    latent_layer = None
    for layer in autoencoder.layers:
        shape = _get_layer_output_shape(layer)
        if shape is not None and len(shape) == 2 and shape[-1] == latent_dim:
            latent_layer = layer
    if latent_layer is not None:
        return keras.Model(autoencoder.input, latent_layer.output)

    layers = list(autoencoder.layers)
    decoder_start_indices = []
    for index, layer in enumerate(layers):
        shape = _get_layer_input_shape(layer)
        if shape is not None and len(shape) == 2 and shape[-1] == latent_dim:
            decoder_start_indices.append(index)
    if decoder_start_indices:
        encoder_output_layer = layers[max(decoder_start_indices) - 1]
        return keras.Model(autoencoder.input, encoder_output_layer.output)

    raise ValueError(f"Could not locate a latent layer with dimension {latent_dim}.")


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
