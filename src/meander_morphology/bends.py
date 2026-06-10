from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .curvature import compute_centerline_curvature, detect_apex_indices, detect_inflection_points
from .geometry import cumulative_distance, maturity_index, normalise_bend, resample_polyline, sinuosity


EndpointMode = Literal["ignore", "auto", "include"]


@dataclass(slots=True)
class Bend:
    """Single meander bend bounded by consecutive curvature inflection points."""

    bend_id: int
    start_index: int
    end_index: int
    apex_index: int
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    curvature: np.ndarray
    raw_x: np.ndarray
    raw_y: np.ndarray
    raw_s: np.ndarray
    width: float | None
    sinuosity: float
    maturity_index: float
    chord_length: float
    chord_widths: float | None
    uses_endpoint_boundary: bool = False

    @property
    def is_edge_bend(self) -> bool:
        """Backward-compatible flag for intervals touching the file boundary."""
        return self.uses_endpoint_boundary

    def metadata(self) -> dict:
        row = asdict(self)
        for key in ["x", "y", "s", "curvature", "raw_x", "raw_y", "raw_s"]:
            row.pop(key, None)
        row["is_edge_bend"] = row.pop("uses_endpoint_boundary")
        return row


def _width_on_resampled_centerline(
    original_x: np.ndarray,
    original_y: np.ndarray,
    width: np.ndarray,
    s_resampled: np.ndarray,
) -> np.ndarray:
    original_s = cumulative_distance(original_x, original_y)
    return np.interp(s_resampled, original_s, np.asarray(width, dtype=float))


def _endpoint_is_plausible_inflection(
    curvature: np.ndarray,
    index: int,
    *,
    tolerance_fraction: float,
) -> bool:
    """Return True when an endpoint curvature is small relative to the reach."""
    max_abs = float(np.nanmax(np.abs(curvature)))
    if not np.isfinite(max_abs) or max_abs == 0:
        return False
    return abs(float(curvature[index])) <= tolerance_fraction * max_abs


def _build_boundaries(
    curvature: np.ndarray,
    true_inflections: np.ndarray,
    *,
    endpoint_mode: EndpointMode,
    endpoint_curvature_tolerance: float,
    include_edge_bends: bool,
) -> np.ndarray:
    """Build extraction boundaries without thinning interior inflections."""
    if include_edge_bends:
        endpoint_mode = "include"

    points = list(np.asarray(true_inflections, dtype=int))
    if endpoint_mode not in {"ignore", "auto", "include"}:
        raise ValueError("endpoint_mode must be one of: ignore, auto, include")

    if endpoint_mode == "include":
        points = [0, *points, len(curvature) - 1]
    elif endpoint_mode == "auto":
        if _endpoint_is_plausible_inflection(
            curvature, 0, tolerance_fraction=endpoint_curvature_tolerance
        ):
            points = [0, *points]
        if _endpoint_is_plausible_inflection(
            curvature, len(curvature) - 1, tolerance_fraction=endpoint_curvature_tolerance
        ):
            points = [*points, len(curvature) - 1]

    return np.unique(np.asarray(points, dtype=int))


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
    endpoint_mode: EndpointMode = "auto",
    endpoint_curvature_tolerance: float = 0.10,
) -> list[Bend]:
    """Extract normalized single bends from a river centerline.

    Interior bends are always cut at consecutive curvature sign-change
    inflection points. Interior inflection points are not removed by spacing,
    because doing so can merge neighbouring single bends into a compound unit.

    ``endpoint_mode`` controls how file endpoints are handled:

    - ``"ignore"``: only true interior sign-change inflections are used.
    - ``"auto"``: an endpoint is used only when its absolute curvature is small
      relative to the reach, making it a plausible boundary inflection.
    - ``"include"``: endpoints are always used, which may include partial edge
      bends.

    ``include_edge_bends=True`` is kept for backwards compatibility and is
    equivalent to ``endpoint_mode="include"``.
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

    true_inflections = detect_inflection_points(curv, s=s, min_spacing=None, include_endpoints=False)
    boundaries = _build_boundaries(
        curv,
        true_inflections,
        endpoint_mode=endpoint_mode,
        endpoint_curvature_tolerance=endpoint_curvature_tolerance,
        include_edge_bends=include_edge_bends,
    )

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

        uses_endpoint = bool(start == 0 or end == len(curv) - 1)
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
                raw_x=seg_x.copy(),
                raw_y=seg_y.copy(),
                raw_s=seg_s.copy(),
                width=bend_width,
                sinuosity=sinuosity(norm_x, norm_y),
                maturity_index=maturity_index(norm_x, norm_y),
                chord_length=chord,
                chord_widths=chord_widths,
                uses_endpoint_boundary=uses_endpoint,
            )
        )
    return bends


def bends_to_metadata_rows(bends: list[Bend]) -> list[dict]:
    """Convert bend objects to CSV-friendly dictionaries."""
    return [bend.metadata() for bend in bends]
