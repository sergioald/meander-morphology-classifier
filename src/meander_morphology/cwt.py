from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def mexican_hat(points: np.ndarray, scale: float) -> np.ndarray:
    """Evaluate a normalized Mexican-hat wavelet."""
    u = np.asarray(points, dtype=float) / float(scale)
    norm = 2.0 / (np.sqrt(3.0 * scale) * np.pi ** 0.25)
    return norm * (1.0 - u**2) * np.exp(-(u**2) / 2.0)


def cwt_mexican_hat(signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Compute a Mexican-hat CWT using convolution."""
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


def mirror_pad_signal(signal: np.ndarray, *, pad_fraction: float = 0.5) -> tuple[np.ndarray, slice]:
    """Mirror-pad a one-dimensional bend signal and return the center crop slice.

    The CWT is computed on the padded signal, then cropped back to the original
    single bend. This reduces cone-of-influence edge artefacts without using
    neighbouring bends.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 4 or pad_fraction <= 0:
        return signal.copy(), slice(0, n)
    pad = max(1, min(n - 1, int(round(n * pad_fraction))))
    left = signal[1:pad + 1][::-1]
    right = signal[-pad - 1:-1][::-1]
    padded = np.concatenate([left, signal, right])
    return padded, slice(pad, pad + n)


def cwt_energy(
    curvature: np.ndarray,
    *,
    n_scales: int = 200,
    max_scale_fraction: float = 0.50,
    pad: bool = True,
    pad_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CWT energy and scales for one isolated bend.

    Scales are based on the original bend length, not the padded length. The
    returned energy is cropped to the original bend extent when ``pad=True``.
    """
    curvature = np.asarray(curvature, dtype=float)
    n = len(curvature)
    if n < 4:
        raise ValueError("curvature must contain at least four points")
    max_scale = max(2.0, n * float(max_scale_fraction))
    scales = np.linspace(1.0, max_scale, n_scales)

    if pad:
        work_signal, crop = mirror_pad_signal(curvature, pad_fraction=pad_fraction)
        coeffs = cwt_mexican_hat(work_signal, scales)[:, crop]
    else:
        coeffs = cwt_mexican_hat(curvature, scales)
    return coeffs**2, scales


def spectrum_image(
    curvature: np.ndarray,
    *,
    image_size: int = 64,
    n_scales: int = 200,
    normalize: bool = True,
    pad: bool = True,
    pad_fraction: float = 0.5,
    max_scale_fraction: float = 0.50,
) -> np.ndarray:
    """Convert one single-bend curvature signal into a square CWT-energy image.

    The default mirror-padding is used only to reduce boundary artefacts. The
    saved image is cropped back to the selected single bend, so adjacent bends
    are not part of the spectrum.
    """
    energy, _ = cwt_energy(
        curvature,
        n_scales=n_scales,
        max_scale_fraction=max_scale_fraction,
        pad=pad,
        pad_fraction=pad_fraction,
    )
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
