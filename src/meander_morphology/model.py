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
        try:
            return tuple(None if dim is None else int(dim) for dim in shape.as_list())
        except Exception:
            return None


def _get_layer_output_shape(layer):
    for attr in ("output_shape", "batch_output_shape"):
        shape = getattr(layer, attr, None)
        parsed = _shape_tuple(shape)
        if parsed is not None:
            return parsed
    try:
        return _shape_tuple(layer.output.shape)
    except Exception:
        return None


def _get_layer_input_shape(layer):
    for attr in ("input_shape", "batch_input_shape"):
        shape = getattr(layer, attr, None)
        parsed = _shape_tuple(shape)
        if parsed is not None:
            return parsed
    try:
        return _shape_tuple(layer.input.shape)
    except Exception:
        return None


def _shape_has_latent_dim(shape, latent_dim: int) -> bool:
    return shape is not None and len(shape) == 2 and shape[-1] == int(latent_dim)


def _nested_model_looks_like_encoder(layer, latent_dim: int) -> bool:
    """Return True when a nested Keras model appears to be the encoder."""
    if not hasattr(layer, "predict"):
        return False

    if _shape_has_latent_dim(_get_layer_output_shape(layer), latent_dim):
        return True

    inner_layers = list(getattr(layer, "layers", []) or [])
    if not inner_layers:
        return False

    last_shape = _get_layer_output_shape(inner_layers[-1])
    return _shape_has_latent_dim(last_shape, latent_dim)


def _model_inputs(model):
    inputs = getattr(model, "inputs", None)
    if inputs:
        return inputs
    try:
        return model.input
    except Exception as exc:
        raise ValueError(
            "Could not determine the autoencoder input tensor. The model may not be built."
        ) from exc


def build_encoder_from_autoencoder(autoencoder, *, latent_dim: int = 2):
    """Build or recover an encoder from a Keras autoencoder.

    The Zenodo autoencoder can load as either:

    * a Functional model containing a nested ``Sequential`` encoder and decoder;
    * a flatter graph where the bottleneck is an explicit layer;
    * a legacy model whose layer shape metadata is incomplete.

    This helper first returns a nested encoder model if one exists. It then
    searches for an explicit bottleneck layer. Finally it falls back to the
    legacy-script strategy: find the first decoder layer whose input is
    ``(batch, latent_dim)`` and return the layer immediately before it.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("TensorFlow/Keras is required for encoder extraction.") from exc

    layers = list(getattr(autoencoder, "layers", []) or [])
    if not layers:
        raise ValueError("The loaded autoencoder has no visible layers.")

    for layer in layers:
        if layer is autoencoder:
            continue
        if _nested_model_looks_like_encoder(layer, int(latent_dim)):
            return layer

    latent_layer = None
    for layer in layers:
        shape = _get_layer_output_shape(layer)
        if _shape_has_latent_dim(shape, int(latent_dim)):
            latent_layer = layer
    if latent_layer is not None:
        return keras.Model(_model_inputs(autoencoder), latent_layer.output)

    decoder_start_indices: list[int] = []
    for index, layer in enumerate(layers):
        shape = _get_layer_input_shape(layer)
        if _shape_has_latent_dim(shape, int(latent_dim)):
            decoder_start_indices.append(index)
    if decoder_start_indices:
        decoder_start = min(decoder_start_indices)
        if decoder_start <= 0:
            raise ValueError("Found decoder start at the first layer; cannot infer encoder output.")
        encoder_output_layer = layers[decoder_start - 1]
        return keras.Model(_model_inputs(autoencoder), encoder_output_layer.output)

    layer_summary = []
    for i, layer in enumerate(layers):
        layer_summary.append(
            f"{i}: {layer.__class__.__name__} name={getattr(layer, 'name', '?')} "
            f"input={_get_layer_input_shape(layer)} output={_get_layer_output_shape(layer)}"
        )
    raise ValueError(
        f"Could not locate a latent bottleneck with dimension {latent_dim}. "
        "Visible layers were:\n" + "\n".join(layer_summary)
    )


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
