"""Curvature-based meander bend detection and classification tools."""

from .bends import Bend, extract_single_bends
from .compound import CompoundBend, extract_compound_bends
from .curvature import compute_centerline_curvature, detect_inflection_points
from .geometry import cumulative_distance, maturity_index, resample_polyline, sinuosity
from .synthetic import KinoshitaParameters, SyntheticBend, generate_kinoshita_bends

__all__ = [
    "Bend",
    "CompoundBend",
    "compute_centerline_curvature",
    "cumulative_distance",
    "detect_inflection_points",
    "extract_compound_bends",
    "extract_single_bends",
    "maturity_index",
    "resample_polyline",
    "sinuosity",
    "KinoshitaParameters",
    "SyntheticBend",
    "generate_kinoshita_bends",
]

__version__ = "0.1.0"
