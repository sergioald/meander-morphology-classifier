from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CompoundLatentResult:
    """Latent coordinates and optional merged per-unit metadata."""

    latent: np.ndarray
    table: pd.DataFrame
    latent_path: Path
    table_path: Path


def prepare_spectra_batch(spectra: np.ndarray) -> np.ndarray:
    """Return spectra as a float32 Keras-style batch ``(N, H, W, 1)``.

    The compound extraction workflow writes ``compound_spectra.npy`` as
    ``(N, H, W)``. Keras image models normally expect an explicit channel
    dimension, so this helper adds it when needed and validates common mistakes.
    """
    arr = np.asarray(spectra, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :, None]
    elif arr.ndim == 3:
        arr = arr[:, :, :, None]
    elif arr.ndim == 4:
        pass
    else:
        raise ValueError("spectra must have shape (H, W), (N, H, W), or (N, H, W, C).")

    if arr.shape[-1] != 1:
        raise ValueError("spectra must have one image channel.")
    if arr.shape[0] == 0:
        raise ValueError("spectra batch is empty.")
    return arr


def _looks_like_encoder(model: object, *, latent_dim: int = 2) -> bool:
    output_shape = getattr(model, "output_shape", None)
    if output_shape is None:
        return False
    if isinstance(output_shape, list):
        if not output_shape:
            return False
        output_shape = output_shape[0]
    try:
        shape = tuple(output_shape)
    except TypeError:
        return False
    return len(shape) == 2 and shape[-1] == latent_dim


def _extract_encoder_from_autoencoder(
    autoencoder: object,
    *,
    latent_layer_name: str = "Latent_Space",
    latent_dim: int = 2,
):
    """Extract an encoder model from a loaded full autoencoder.

    The compound paper/package uses a latent layer called ``Latent_Space``.
    The repository also contains a robust fallback helper for the published
    single-bend model, so this function tries both approaches.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - exercised only without TensorFlow
        raise ImportError("TensorFlow is required to extract an encoder from a full model.") from exc

    if _looks_like_encoder(autoencoder, latent_dim=latent_dim):
        return autoencoder

    if latent_layer_name:
        try:
            latent_layer = autoencoder.get_layer(latent_layer_name)
            return tf.keras.Model(
                inputs=autoencoder.input,
                outputs=latent_layer.output,
                name="compound_encoder",
            )
        except Exception:
            pass

    try:
        from .model import build_encoder_from_autoencoder

        return build_encoder_from_autoencoder(autoencoder, latent_dim=latent_dim)
    except Exception as exc:
        raise ValueError(
            "Could not extract a latent encoder from the supplied autoencoder. "
            f"Tried layer '{latent_layer_name}' and the repository fallback helper."
        ) from exc


def load_compound_encoder(
    model_path: str | Path,
    *,
    model_is_encoder: bool = False,
    latent_layer_name: str = "Latent_Space",
    latent_dim: int = 2,
):
    """Load an encoder, or load a full autoencoder and extract its encoder.

    Parameters
    ----------
    model_path:
        Path to ``encoder_only.keras``, ``encoder_only.h5``, or a full trained
        autoencoder such as ``trained_autoencoder.h5``.
    model_is_encoder:
        Set True when ``model_path`` already points to an encoder-only model.
    latent_layer_name:
        Name of the latent layer to use when loading a full autoencoder.
    latent_dim:
        Expected dimensionality of the latent space.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TensorFlow is required for model encoding. Install with: "
            "python -m pip install -e \".[deep-learning]\""
        ) from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if model_is_encoder:
        return tf.keras.models.load_model(model_path, compile=False)

    try:
        from .model import load_autoencoder

        autoencoder = load_autoencoder(model_path)
    except Exception:
        autoencoder = tf.keras.models.load_model(model_path, compile=False)

    return _extract_encoder_from_autoencoder(
        autoencoder,
        latent_layer_name=latent_layer_name,
        latent_dim=latent_dim,
    )


def encode_spectra_with_encoder(
    encoder: object,
    spectra: np.ndarray,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode spectra with any object exposing a Keras-like ``predict`` method."""
    batch = prepare_spectra_batch(spectra)
    if not hasattr(encoder, "predict"):
        raise TypeError("encoder must expose a predict(...) method.")
    latent = encoder.predict(batch, batch_size=int(batch_size), verbose=0)
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2:
        latent = latent.reshape((latent.shape[0], -1))
    if latent.shape[0] != batch.shape[0]:
        raise ValueError("encoder returned a different number of rows than the input spectra.")
    return latent


def encode_compound_spectra(
    spectra_path: str | Path,
    model_path: str | Path,
    *,
    model_is_encoder: bool = False,
    latent_layer_name: str = "Latent_Space",
    latent_dim: int = 2,
    batch_size: int = 128,
) -> np.ndarray:
    """Load ``compound_spectra.npy`` and return latent coordinates."""
    spectra = np.load(Path(spectra_path))
    encoder = load_compound_encoder(
        model_path,
        model_is_encoder=model_is_encoder,
        latent_layer_name=latent_layer_name,
        latent_dim=latent_dim,
    )
    return encode_spectra_with_encoder(encoder, spectra, batch_size=batch_size)


def _latent_dataframe(latent: np.ndarray, summary_path: str | Path | None = None) -> pd.DataFrame:
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2:
        raise ValueError("latent must be a two-dimensional array.")

    columns = {f"latent_{i + 1}": latent[:, i] for i in range(latent.shape[1])}
    table = pd.DataFrame(columns)
    table.insert(0, "unit_id", np.arange(latent.shape[0], dtype=int))

    if summary_path is None:
        return table

    summary = pd.read_csv(summary_path)
    if "unit_id" not in summary.columns:
        raise ValueError("summary CSV must contain a unit_id column.")
    return summary.merge(table, on="unit_id", how="left", validate="one_to_one")


def save_latent_outputs(
    latent: np.ndarray,
    output_dir: str | Path,
    *,
    summary_path: str | Path | None = None,
    prefix: str = "compound",
) -> CompoundLatentResult:
    """Save latent coordinates to ``.npy`` and CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent = np.asarray(latent, dtype=float)
    latent_path = output_dir / f"{prefix}_latent.npy"
    table_path = output_dir / f"{prefix}_latent.csv"

    np.save(latent_path, latent)
    table = _latent_dataframe(latent, summary_path=summary_path)
    table.to_csv(table_path, index=False)

    return CompoundLatentResult(
        latent=latent,
        table=table,
        latent_path=latent_path,
        table_path=table_path,
    )


def save_latent_plot(
    latent: np.ndarray,
    output_path: str | Path,
    *,
    background_latent_path: str | Path | None = None,
    title: str = "Compound-bend latent space",
) -> Path:
    """Save a simple latent-space scatter plot.

    The plot is intentionally dependency-light and optional. It supports the
    background world cloud from the package/Zenodo as a grey reference layer.
    """
    import matplotlib.pyplot as plt

    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2 or latent.shape[1] < 2:
        raise ValueError("latent must have at least two columns to make a 2-D plot.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    if background_latent_path is not None:
        background = np.load(Path(background_latent_path))
        background = np.asarray(background, dtype=float)
        if background.ndim == 2 and background.shape[1] >= 2:
            ax.scatter(background[:, 0], background[:, 1], s=2, alpha=0.15, label="background")

    ax.scatter(latent[:, 0], latent[:, 1], s=36, label="compound units")
    for i, (x, y) in enumerate(latent[:, :2]):
        ax.annotate(str(i), (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("latent_1")
    ax.set_ylabel("latent_2")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
