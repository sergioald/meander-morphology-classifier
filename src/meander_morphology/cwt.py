from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
from skimage.restoration import denoise_wavelet, estimate_sigma

try:  # PyWavelets matches the original research scripts when installed.
    import pywt
except Exception:  # pragma: no cover - exercised only when PyWavelets is absent
    pywt = None


@dataclass(slots=True)
class SpectrumResult:
    """CWT energy spectrum and axis metadata."""

    energy: np.ndarray
    periods: np.ndarray
    s: np.ndarray
    curvature: np.ndarray


def _odd_window(length: int, fraction: float = 0.05, minimum: int = 3) -> int:
    """Return a valid odd Savitzky-Golay window length."""
    if length < 3:
        return 3
    window = max(minimum, int(round(length * fraction)))
    if window % 2 == 0:
        window += 1
    max_valid = length if length % 2 == 1 else length - 1
    return max(3, min(window, max_valid))


def _safe_denoise(values: np.ndarray, *, wavelet: str = "sym9", sigma_factor: float = 1.0) -> np.ndarray:
    """Wavelet-denoise a 1-D vector, falling back to the original vector if needed."""
    values = np.asarray(values, dtype=float)
    if values.size < 8:
        return values.copy()
    try:
        sigma = estimate_sigma(values, average_sigmas=True)
        return np.asarray(
            denoise_wavelet(
                values,
                method="VisuShrink",
                mode="soft",
                wavelet=wavelet,
                sigma=float(sigma) * float(sigma_factor),
                rescale_sigma=True,
            ),
            dtype=float,
        )
    except Exception:
        return values.copy()


def _cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean distance along a polyline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size == 0:
        return np.asarray([], dtype=float)
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])


def _resample_xy(x: np.ndarray, y: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a polyline to ``n_points`` along cumulative distance."""
    s = _cumulative_distance(x, y)
    if s.size < 2 or s[-1] <= 0:
        raise ValueError("polyline must contain at least two distinct points")
    target = np.linspace(0.0, s[-1], int(n_points))
    return target, np.interp(target, s, x), np.interp(target, s, y)


def reflect_bend_geometry(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, slice]:
    """Reflect an isolated bend at both ends and return the central crop slice.

    This reproduces the intent of the original scripts: create artificial
    mirrored neighbours only to stabilise derivatives near the bend boundaries.
    The mirrored sections are removed before the CWT is computed.
    """
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.shape != yy.shape:
        raise ValueError("x and y must have the same shape")
    if xx.size < 4:
        raise ValueError("bend geometry must contain at least four points")

    if xx[-1] < xx[0]:
        xx = xx[::-1]
        yy = yy[::-1]

    xi = xx - xx[0]
    yi = yy - yy[0]
    xf = xx - xx[-1]
    yf = yy - yy[-1]

    left_x = -xi + xx[0]
    left_y = -yi + yy[0]
    right_x = -xf + xx[-1]
    right_y = -yf + yy[-1]

    if left_x[-1] < left_x[0]:
        left_x = left_x[::-1]
        left_y = left_y[::-1]
    if right_x[-1] < right_x[0]:
        right_x = right_x[::-1]
        right_y = right_y[::-1]

    padded_x = np.concatenate([left_x, xx, right_x])
    padded_y = np.concatenate([left_y, yy, right_y])
    n = xx.size
    return padded_x, padded_y, slice(n, 2 * n)


def legacy_single_bend_curvature(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_points: int = 201,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized single-bend geometry and curvature using the legacy workflow.

    The original scripts mirror the selected bend in x/y, compute angle and
    curvature on the three-bend reflected geometry, then crop back to the
    central isolated bend before computing the CWT. This function implements
    that order of operations.

    Returns
    -------
    s, x_center, y_center, curvature_center
        Central-bend distance coordinate, normalized geometry and curvature.
    """
    n = int(target_points)
    if n < 16:
        raise ValueError("target_points must be at least 16")

    # Resample the selected bend to a fixed length first. This mirrors the old
    # code path where the central crop was 201 samples.
    _, bx, by = _resample_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float), n)
    px, py, center = reflect_bend_geometry(bx, by)

    # Resample the reflected triplet to exactly 3*n samples so that the center
    # slice is unambiguous and comparable with the old cut_st:2*cut_st crop.
    s_full, px, py = _resample_xy(px, py, 3 * n)

    if smooth:
        win = _odd_window(py.size, fraction=0.05)
        px = savgol_filter(px, win, 1)
        py = savgol_filter(py, win, 1)
        px = _safe_denoise(px, wavelet="sym9", sigma_factor=2.0)
        py = _safe_denoise(py, wavelet="sym9", sigma_factor=2.0)

    dx = np.gradient(px, edge_order=1)
    dy = np.gradient(py, edge_order=1)

    if smooth:
        dx = _safe_denoise(dx, wavelet="sym9", sigma_factor=6.0)
        dy = _safe_denoise(dy, wavelet="sym9", sigma_factor=6.0)

    angle = -np.unwrap(np.arctan2(dy, dx))
    if smooth:
        angle = _safe_denoise(angle, wavelet="sym9", sigma_factor=5.0)

    curvature = np.gradient(angle, edge_order=1)
    if smooth:
        curvature = _safe_denoise(curvature, wavelet="sym9", sigma_factor=5.0)
        curvature = savgol_filter(curvature, _odd_window(curvature.size, fraction=0.05), 1)
        curvature = _safe_denoise(curvature, wavelet="sym9", sigma_factor=5.0)

    center = slice(n, 2 * n)
    s = s_full[center]
    s = s - s[0]
    if s[-1] > 0:
        s = s / s[-1]

    return s, px[center], py[center], curvature[center]


def mexican_hat(points: np.ndarray, scale: float) -> np.ndarray:
    """Evaluate a normalized Mexican-hat wavelet for the fallback CWT."""
    u = np.asarray(points, dtype=float) / float(scale)
    norm = 2.0 / (np.sqrt(3.0 * scale) * np.pi ** 0.25)
    return norm * (1.0 - u**2) * np.exp(-(u**2) / 2.0)


def _fallback_cwt_mexican_hat(signal: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Small dependency-light Mexican-hat CWT fallback."""
    signal = np.asarray(signal, dtype=float) - float(np.nanmean(signal))
    coeffs = []
    n = signal.size
    for period in np.asarray(periods, dtype=float):
        # For display and tests this approximate mapping is sufficient when
        # PyWavelets is unavailable. The preferred path below uses pywt.cwt.
        scale = max(1.0, period / 4.0)
        half_width = max(4, int(np.ceil(6 * scale)))
        t = np.arange(-half_width, half_width + 1)
        wavelet = mexican_hat(t, scale)
        conv = np.convolve(signal, wavelet[::-1], mode="same")
        if conv.size != n:
            start = (conv.size - n) // 2
            conv = conv[start : start + n]
        coeffs.append(conv)
    return np.asarray(coeffs)


def cwt_energy(
    curvature: np.ndarray,
    *,
    wavelet: str = "mexh",
    log_energy: bool = False,
) -> SpectrumResult:
    """Compute CWT energy from an already isolated bend-curvature signal.

    This intentionally does **not** add neighbouring bends or mirror padding at
    the CWT stage. That matches the original scripts, which crop the isolated
    bend first and then call ``pywt.cwt`` on the cropped curvature vector.
    """
    dcg = np.asarray(curvature, dtype=float)
    if dcg.size < 4:
        raise ValueError("curvature must contain at least four points")

    n = dcg.size
    s = np.linspace(0.0, 1.0, n)
    dt = 1.0 / n
    fs = 1.0 / dt
    frequencies = np.arange(1, n, dtype=float) / fs
    periods = 1.0 / frequencies

    if pywt is not None:
        scales = pywt.frequency2scale(wavelet, frequencies)
        coeffs, _ = pywt.cwt(dcg, scales=scales, wavelet=wavelet, sampling_period=dt)
    else:
        coeffs = _fallback_cwt_mexican_hat(dcg, periods)

    energy = np.abs(coeffs) ** 2
    if log_energy:
        energy = np.log(energy + 1e-10)
    return SpectrumResult(energy=energy, periods=periods, s=s, curvature=dcg)


def cwt_energy_from_geometry(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_points: int = 201,
    wavelet: str = "mexh",
    log_energy: bool = False,
    smooth: bool = True,
) -> SpectrumResult:
    """Compute the legacy single-bend CWT spectrum from isolated bend geometry."""
    s, _, _, curvature = legacy_single_bend_curvature(
        x,
        y,
        target_points=target_points,
        smooth=smooth,
    )
    result = cwt_energy(curvature, wavelet=wavelet, log_energy=log_energy)
    return SpectrumResult(
        energy=result.energy,
        periods=result.periods,
        s=s,
        curvature=curvature,
    )


def spectrum_image(
    curvature: np.ndarray,
    *,
    image_size: int = 64,
    normalize: bool = True,
    log_energy: bool = False,
    **_: object,
) -> np.ndarray:
    """Convert an isolated curvature signal into a square CWT-energy image.

    Extra keyword arguments are accepted for backward compatibility with older
    scripts that passed CWT-padding options. They are ignored because the fixed
    workflow crops to one bend before CWT instead of padding during CWT.
    """
    result = cwt_energy(curvature, log_energy=log_energy)
    return energy_to_image(result.energy, image_size=image_size, normalize=normalize)


def spectrum_image_from_geometry(
    x: np.ndarray,
    y: np.ndarray,
    *,
    image_size: int = 64,
    target_points: int = 201,
    normalize: bool = True,
    log_energy: bool = False,
    smooth: bool = True,
) -> np.ndarray:
    """Convert isolated bend geometry into a legacy-compatible CWT-energy image."""
    result = cwt_energy_from_geometry(
        x,
        y,
        target_points=target_points,
        log_energy=log_energy,
        smooth=smooth,
    )
    return energy_to_image(result.energy, image_size=image_size, normalize=normalize)


def energy_to_image(energy: np.ndarray, *, image_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Resize a CWT-energy matrix to a square image."""
    image = np.asarray(energy, dtype=float)
    if normalize:
        finite = np.isfinite(image)
        if finite.any():
            image = image.copy()
            lo = float(np.nanmin(image[finite]))
            hi = float(np.nanmax(image[finite]))
            if hi > lo:
                image = (image - lo) / (hi - lo)
            else:
                image = np.zeros_like(image)
    zoom_factors = (image_size / image.shape[0], image_size / image.shape[1])
    resized = zoom(image, zoom_factors, order=1)
    return np.clip(resized, 0.0, 1.0)


def save_spectrum_image(path: str, image: np.ndarray) -> None:
    """Save a spectrum image as PNG."""
    import matplotlib.pyplot as plt

    plt.imsave(path, image, cmap="gray", vmin=0.0, vmax=1.0)
