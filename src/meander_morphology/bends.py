from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .curvature import compute_centerline_curvature, detect_apex_indices, detect_inflection_points
from .geometry import cumulative_distance, maturity_index, normalise_bend, resample_polyline, sinuosity


@dataclass(slots=True)
class Bend:
    """Single meander bend bounded by consecutive inflection points."""

    bend_id: int
    start_index: int
    end_index: int
    apex_index: int
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    curvature: np.ndarray
    width: float | None
    sinuosity: float
    maturity_index: float
    chord_length: float

    def metadata(self) -> dict:
        row = asdict(self)
        for key in ["x", "y", "s", "curvature"]:
            row.pop(key, None)
        return row


def extract_single_bends(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: float | np.ndarray | None = None,
    points_per_width: int = 25,
    min_spacing_widths: float = 6.0,
    min_abs_curvature: float | None = None,
    bend_points: int = 201,
) -> list[Bend]:
    """Extract normalized single bends from a river centerline.

    The workflow mirrors the research scripts: resample/smooth the centerline,
    compute curvature, detect inflection points, filter very short bends and rotate
    each extracted bend into a common frame.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if width is None:
        mean_width = None
    elif np.ndim(width) == 0:
        mean_width = float(width)
    else:
        mean_width = float(np.nanmean(width))

    if mean_width and mean_width > 0:
        approx_length = cumulative_distance(x, y)[-1]
        resample_points = max(50, int(points_per_width * approx_length / mean_width))
    else:
        resample_points = max(200, len(x))

    s, xs, ys, curv = compute_centerline_curvature(x, y, resample_points=resample_points)
    min_spacing = min_spacing_widths * mean_width if mean_width else None
    inflections = detect_inflection_points(curv, s=s, min_spacing=min_spacing, include_endpoints=True)
    apexes = detect_apex_indices(curv, inflections)

    bends: list[Bend] = []
    for bend_id, (start, end, apex) in enumerate(zip(inflections[:-1], inflections[1:], apexes)):
        if end <= start + 2:
            continue
        if min_abs_curvature is not None and abs(curv[apex]) < min_abs_curvature:
            continue

        seg_x = xs[start:end + 1]
        seg_y = ys[start:end + 1]
        seg_s = s[start:end + 1] - s[start]
        seg_c = curv[start:end + 1]

        bend_width = mean_width
        if np.ndim(width) == 1:
            original_s = cumulative_distance(x, y)
            width_resampled = np.interp(s, original_s, np.asarray(width, dtype=float))
            bend_width = float(np.nanmean(width_resampled[start:end + 1]))

        norm_x, norm_y = normalise_bend(seg_x, seg_y, bend_width)
        norm_x, norm_y, norm_s = resample_polyline(norm_x, norm_y, n_points=bend_points)
        norm_c = np.interp(norm_s, np.linspace(norm_s.min(), norm_s.max(), len(seg_c)), seg_c)
        chord = float(np.hypot(norm_x[-1] - norm_x[0], norm_y[-1] - norm_y[0]))

        bends.append(
            Bend(
                bend_id=len(bends),
                start_index=int(start),
                end_index=int(end),
                apex_index=int(apex),
                x=norm_x,
                y=norm_y,
                s=norm_s,
                curvature=norm_c,
                width=bend_width,
                sinuosity=sinuosity(norm_x, norm_y),
                maturity_index=maturity_index(norm_x, norm_y),
                chord_length=chord,
            )
        )
    return bends


def bends_to_metadata_rows(bends: list[Bend]) -> list[dict]:
    """Convert bend objects to CSV-friendly dictionaries."""
    return [bend.metadata() for bend in bends]
