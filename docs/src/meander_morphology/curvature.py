from __future__ import annotations

import numpy as np

from .geometry import cumulative_distance, resample_polyline
from .preprocessing import smooth_centerline, smooth_signal


def compute_centerline_curvature(
    x: np.ndarray,
    y: np.ndarray,
    *,
    resample_points: int | None = None,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute arclength, smoothed coordinates and curvature of a centerline.

    Curvature is computed as d(theta)/ds, where theta is the unwrapped tangent angle.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if resample_points is not None:
        x, y, _ = resample_polyline(x, y, n_points=resample_points)
    if smooth:
        x, y = smooth_centerline(x, y)
    s = cumulative_distance(x, y)
    dx = np.gradient(x, s, edge_order=1)
    dy = np.gradient(y, s, edge_order=1)
    theta = np.unwrap(np.arctan2(dy, dx))
    if smooth:
        theta = smooth_signal(theta, window_fraction=0.03)
    curvature = np.gradient(theta, s, edge_order=1)
    if smooth:
        curvature = smooth_signal(curvature, window_fraction=0.03)
    return s, x, y, curvature


def detect_inflection_points(
    curvature: np.ndarray,
    *,
    s: np.ndarray | None = None,
    min_spacing: float | None = None,
    include_endpoints: bool = True,
) -> np.ndarray:
    """Detect curvature sign-change indices and optionally filter by spacing."""
    curvature = np.asarray(curvature, dtype=float)
    if curvature.ndim != 1:
        raise ValueError("curvature must be one-dimensional.")
    signs = np.sign(curvature)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    changes = np.flatnonzero(signs[:-1] * signs[1:] < 0) + 1
    points = changes.tolist()
    if include_endpoints:
        points = [0, *points, len(curvature) - 1]
    if min_spacing is not None and s is not None and points:
        s = np.asarray(s, dtype=float)
        filtered = [points[0]]
        for idx in points[1:]:
            if s[idx] - s[filtered[-1]] >= min_spacing:
                filtered.append(idx)
        if filtered[-1] != points[-1] and include_endpoints:
            if s[points[-1]] - s[filtered[-1]] >= min_spacing:
                filtered.append(points[-1])
            else:
                filtered[-1] = points[-1]
        points = filtered
    return np.asarray(points, dtype=int)


def detect_apex_indices(curvature: np.ndarray, inflection_indices: np.ndarray) -> np.ndarray:
    """Return the maximum absolute-curvature index inside each bend interval."""
    curvature = np.asarray(curvature, dtype=float)
    apex = []
    for start, end in zip(inflection_indices[:-1], inflection_indices[1:]):
        if end <= start + 1:
            apex.append(start)
            continue
        local = np.abs(curvature[start:end + 1])
        apex.append(start + int(np.nanargmax(local)))
    return np.asarray(apex, dtype=int)
