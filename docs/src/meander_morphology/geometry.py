from __future__ import annotations

import numpy as np


def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return cumulative along-centerline distance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if len(x) < 2:
        raise ValueError("At least two points are required.")
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(ds)])


def resample_polyline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_points: int | None = None,
    spacing: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a polyline by arclength interpolation."""
    if n_points is None and spacing is None:
        raise ValueError("Provide either n_points or spacing.")
    s = cumulative_distance(x, y)
    length = float(s[-1])
    if length == 0:
        raise ValueError("Cannot resample a zero-length polyline.")
    if n_points is None:
        n_points = max(2, int(np.ceil(length / float(spacing))) + 1)
    s_new = np.linspace(0.0, length, int(n_points))
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new, s_new


def sinuosity(x: np.ndarray, y: np.ndarray) -> float:
    """Return intrinsic length divided by end-to-end distance."""
    s = cumulative_distance(x, y)
    chord = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if chord == 0:
        return float("nan")
    return float(s[-1] / chord)


def maturity_index(x: np.ndarray, y: np.ndarray) -> float:
    """Return amplitude-to-chord ratio used as maturity index."""
    chord = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if chord == 0:
        return float("nan")
    amplitude = float(np.nanmax(y) - np.nanmin(y))
    return amplitude / chord


def rotate_to_x_axis(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate the first point to the origin and rotate the last point onto +x."""
    x = np.asarray(x, dtype=float) - float(x[0])
    y = np.asarray(y, dtype=float) - float(y[0])
    angle = np.arctan2(y[-1], x[-1])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    xr = cos_a * x - sin_a * y
    yr = sin_a * x + cos_a * y
    if np.nanmax(yr) < abs(np.nanmin(yr)):
        yr = -yr
    if xr[-1] < xr[0]:
        xr = xr[::-1]
        yr = yr[::-1]
        xr = xr - xr[0]
        yr = yr - yr[0]
    return xr, yr


def normalise_bend(x: np.ndarray, y: np.ndarray, width: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a bend by width, align it horizontally and keep it concave upward."""
    scale = 1.0 if width is None else float(width)
    if scale <= 0:
        raise ValueError("width must be positive when provided.")
    xr, yr = rotate_to_x_axis(np.asarray(x, dtype=float) / scale, np.asarray(y, dtype=float) / scale)
    return xr, yr
