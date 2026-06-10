from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .curvature import compute_centerline_curvature, detect_apex_indices, detect_inflection_points
from .geometry import cumulative_distance, maturity_index, normalise_bend, resample_polyline, sinuosity


@dataclass(slots=True)
class Bend:
    """Single meander bend bounded by consecutive true inflection points."""

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
    chord_widths: float | None
    is_edge_bend: bool = False

    def metadata(self) -> dict:
        row = asdict(self)
        for key in ["x", "y", "s", "curvature"]:
            row.pop(key, None)
        return row


def _width_on_resampled_centerline(
    original_x: np.ndarray,
    original_y: np.ndarray,
    width: np.ndarray,
    s_resampled: np.ndarray,
) -> np.ndarray:
    original_s = cumulative_distance(original_x, original_y)
    return np.interp(s_resampled, original_s, np.asarray(width, dtype=float))


def extract_single_bends(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: float | np.ndarray | None = None,
    points_per_width: int = 25,
    min_spacing_widths: float | None = None,
    min_chord_widths: float | None = None,
    min_abs_curvature: float | None = None,
    bend_points: int = 201,
    include_edge_bends: bool = False,
) -> list[Bend]:
    """Extract normalized single bends from a river centerline.

    Single-bend mode must not merge neighbouring bends. The method therefore
    cuts the centerline at **consecutive curvature sign-change inflection
    points**. Short or nearly flat candidates are discarded after extraction
    using bend-level filters.

    Parameters
    ----------
    x, y:
        Centerline coordinates.
    width:
        Constant width or width values along the original centerline.
    points_per_width:
        Resampling density used before curvature estimation.
    min_spacing_widths:
        Backward-compatible alias for ``min_chord_widths``. Older versions used
        this value to remove inflection points before extraction, which could
        merge neighbouring single bends into compound-looking bends.
    min_chord_widths:
        Minimum end-to-end chord length, expressed in channel widths, required
        to retain a bend. The WRR workflow used a 5--8 width range for removing
        spurious short bends; this filter is applied to candidate bends rather
        than to inflection points. Set to ``None`` to disable.
    min_abs_curvature:
        Optional minimum absolute apex curvature.
    bend_points:
        Number of points in each normalized bend.
    include_edge_bends:
        Include partial first/last reaches bounded by a file endpoint and one
        true inflection point. Default is false because these are not complete
        single-lobe bends.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    if min_chord_widths is None and min_spacing_widths is not None:
        min_chord_widths = min_spacing_widths

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

    # Detect true sign-change inflections first. Do not thin these points by
    # distance, because thinning removes interior inflections and creates
    # compound-looking units.
    true_inflections = detect_inflection_points(curv, s=s, min_spacing=None, include_endpoints=False)
    if len(true_inflections) < 2 and not include_edge_bends:
        return []

    if include_edge_bends:
        boundaries = np.unique(np.concatenate([[0], true_inflections, [len(curv) - 1]])).astype(int)
    else:
        boundaries = true_inflections.astype(int)

    if len(boundaries) < 2:
        return []

    width_resampled = None
    if np.ndim(width) == 1:
        width_resampled = _width_on_resampled_centerline(x, y, np.asarray(width, dtype=float), s)

    apexes = detect_apex_indices(curv, boundaries)

    bends: list[Bend] = []
    for start, end, apex in zip(boundaries[:-1], boundaries[1:], apexes):
        if end <= start + 2:
            continue

        is_edge = bool(start == 0 or end == len(curv) - 1)
        bend_width = mean_width
        if width_resampled is not None:
            bend_width = float(np.nanmean(width_resampled[start:end + 1]))

        raw_chord = float(np.hypot(xs[end] - xs[start], ys[end] - ys[start]))
        chord_widths = None
        if bend_width and bend_width > 0:
            chord_widths = raw_chord / bend_width
            if min_chord_widths is not None and chord_widths < min_chord_widths:
                continue

        if min_abs_curvature is not None and abs(curv[apex]) < min_abs_curvature:
            continue

        seg_x = xs[start:end + 1]
        seg_y = ys[start:end + 1]
        seg_s = s[start:end + 1] - s[start]
        seg_c = curv[start:end + 1]

        norm_x, norm_y = normalise_bend(seg_x, seg_y, bend_width)
        norm_x, norm_y, norm_s = resample_polyline(norm_x, norm_y, n_points=bend_points)
        source_s = np.linspace(norm_s.min(), norm_s.max(), len(seg_c))
        norm_c = np.interp(norm_s, source_s, seg_c)
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
                chord_widths=chord_widths,
                is_edge_bend=is_edge,
            )
        )
    return bends


def bends_to_metadata_rows(bends: list[Bend]) -> list[dict]:
    """Convert bend objects to CSV-friendly dictionaries."""
    return [bend.metadata() for bend in bends]
