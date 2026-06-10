from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def mexican_hat(points: np.ndarray, scale: float) -> np.ndarray:
    """Evaluate a normalized Mexican-hat wavelet."""
    u = np.asarray(points, dtype=float) / float(scale)
    norm = 2.0 / (np.sqrt(3.0 * scale) * np.pi ** 0.25)
    return norm * (1.0 - u**2) * np.exp(-(u**2) / 2.0)


def cwt_mexican_hat(signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Compute a simple Mexican-hat CWT using convolution.

    PyWavelets can be installed for production workflows, but this lightweight
    implementation keeps the core package testable without optional dependencies.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)
    coeffs = []
    n = len(signal)
    for scale in np.asarray(scales, dtype=float):
        half_width = max(4, int(np.ceil(6 * scale)))
        t = np.arange(-half_width, half_width + 1)
        wavelet = mexican_hat(t, scale)
        conv = np.convolve(signal, wavelet[::-1], mode="same")
        if len(conv) != n:
            start = (len(conv) - n) // 2
            conv = conv[start:start + n]
        coeffs.append(conv)
    return np.asarray(coeffs)


def cwt_energy(curvature: np.ndarray, *, n_scales: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return CWT energy and scales for a curvature signal."""
    scales = np.linspace(1.0, max(2.0, len(curvature) / 2.0), n_scales)
    coeffs = cwt_mexican_hat(curvature, scales)
    return coeffs**2, scales


def spectrum_image(
    curvature: np.ndarray,
    *,
    image_size: int = 64,
    n_scales: int = 200,
    normalize: bool = True,
) -> np.ndarray:
    """Convert curvature into a square grayscale CWT-energy image."""
    energy, _ = cwt_energy(curvature, n_scales=n_scales)
    image = np.asarray(energy, dtype=float)
    if normalize:
        max_value = float(np.nanmax(image))
        if max_value > 0:
            image = image / max_value
    zoom_factors = (image_size / image.shape[0], image_size / image.shape[1])
    resized = zoom(image, zoom_factors, order=1)
    return np.clip(resized, 0.0, 1.0)


def save_spectrum_image(path: str, image: np.ndarray) -> None:
    """Save a spectrum image as PNG."""
    import matplotlib.pyplot as plt

    plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=1.0)
