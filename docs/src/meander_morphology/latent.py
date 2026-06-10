from __future__ import annotations

import numpy as np

from .model import prepare_images_for_keras


def encode_spectra(encoder, images: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
    """Encode CWT spectrum images into latent coordinates."""
    x = prepare_images_for_keras(images)
    return np.asarray(encoder.predict(x, batch_size=batch_size, verbose=0))


def reconstruct_spectra(autoencoder, images: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
    """Reconstruct spectrum images with an autoencoder."""
    x = prepare_images_for_keras(images)
    return np.asarray(autoencoder.predict(x, batch_size=batch_size, verbose=0))
