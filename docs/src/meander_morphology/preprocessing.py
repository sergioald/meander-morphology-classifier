from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def odd_window_length(length: int, fraction: float = 0.05, minimum: int = 5) -> int:
    """Return a valid odd Savitzky-Golay window length."""
    win = max(minimum, int(round(length * fraction)))
    if win % 2 == 0:
        win += 1
    if win >= length:
        win = length - 1 if (length - 1) % 2 else length - 2
    return max(3, win)


def smooth_signal(values: np.ndarray, *, window_fraction: float = 0.05, polyorder: int = 2) -> np.ndarray:
    """Smooth one-dimensional data with a safe Savitzky-Golay filter."""
    values = np.asarray(values, dtype=float)
    if values.size < 7:
        return values.copy()
    win = odd_window_length(values.size, fraction=window_fraction, minimum=5)
    poly = min(polyorder, win - 1)
    return savgol_filter(values, window_length=win, polyorder=poly, mode="interp")


def smooth_centerline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_fraction: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth x and y coordinates before curvature calculation."""
    return (
        smooth_signal(x, window_fraction=window_fraction),
        smooth_signal(y, window_fraction=window_fraction),
    )
