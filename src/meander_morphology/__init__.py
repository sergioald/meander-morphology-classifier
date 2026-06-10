"""Curvature-based meander bend detection and classification tools."""

from .bends import Bend, extract_single_bends
from .curvature import compute_centerline_curvature, detect_inflection_points
from .geometry import cumulative_distance, maturity_index, resample_polyline, sinuosity

__all__ = [
    "Bend",
    "compute_centerline_curvature",
    "cumulative_distance",
    "detect_inflection_points",
    "extract_single_bends",
    "maturity_index",
    "resample_polyline",
    "sinuosity",
]

__version__ = "0.1.0"
