from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .bends import _width_on_resampled_centerline
from .curvature import compute_centerline_curvature, detect_inflection_points
from .geometry import cumulative_distance, maturity_index, normalise_bend, resample_polyline, sinuosity

try:  # PyWavelets is part of the core project dependencies.
    import pywt
except Exception:  # pragma: no cover - only used when dependency is unavailable
    pywt = None


@dataclass(slots=True)
class CompoundBend:
    """Compound or simple meander unit detected from the reach-scale CWT-energy envelope."""

    unit_id: int
    start_index: int
    end_index: int
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
    length: float
    length_widths: float | None
    n_internal_inflections: int
    n_lobes: int
    is_compound: bool
    boundary_method: str = "cwt_energy_valley"

    def metadata(self) -> dict:
        """Return CSV-friendly unit metadata."""
        row = asdict(self)
        for key in ["x", "y", "s", "curvature", "raw_x", "raw_y", "raw_s"]:
            row.pop(key, None)
        return row


@dataclass(slots=True)
class CompoundSegmentationResult:
    """Diagnostic output from the compound-bend CWT segmentation step."""

    energy: np.ndarray
    frequencies: np.ndarray
    s: np.ndarray
    ridge_indices: np.ndarray
    trough_indices: np.ndarray
    corridor_energy: np.ndarray
    normalised_energy: np.ndarray
    boundary_indices: np.ndarray


def _as_odd_window(samples: int, *, minimum: int = 3) -> int:
    """Return an odd moving-window length valid for a vector of ``samples``."""
    samples = int(samples)
    if samples <= 1:
        return 1
    window = max(int(minimum), samples)
    if window % 2 == 0:
        window += 1
    return max(1, window)


def _median_spacing(s: np.ndarray) -> float:
    ds = np.diff(np.asarray(s, dtype=float))
    ds = ds[np.isfinite(ds) & (ds > 0)]
    if ds.size == 0:
        raise ValueError("s must be strictly increasing over at least two points.")
    return float(np.nanmedian(ds))


def _cwt_reach_energy(
    curvature: np.ndarray,
    s: np.ndarray,
    *,
    wavelet: str = "mexh",
    min_frequency_index: int = 1,
    max_frequency_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reach-scale Mexican-hat CWT energy for segmentation.

    The frequency indexing follows the legacy project convention: normalized
    pseudo-frequencies are converted to PyWavelets scales and evaluated over the
    full reach. The returned frequencies are physical pseudo-frequencies in
    cycles per coordinate unit according to the sampling interval of ``s``.
    """
    if pywt is None:  # pragma: no cover
        raise ImportError("PyWavelets is required for compound-bend segmentation.")

    kappa = np.asarray(curvature, dtype=float)
    if kappa.ndim != 1 or kappa.size < 8:
        raise ValueError("curvature must be a one-dimensional vector with at least 8 samples.")

    ds = _median_spacing(s)
    n = kappa.size
    min_frequency_index = max(1, int(min_frequency_index))
    if max_frequency_index is None:
        max_frequency_index = max(min_frequency_index + 1, n // 2)
    max_frequency_index = min(int(max_frequency_index), n - 1)
    if max_frequency_index <= min_frequency_index:
        max_frequency_index = min(n - 1, min_frequency_index + 1)

    normalised_frequencies = np.arange(min_frequency_index, max_frequency_index + 1, dtype=float) / float(n)
    scales = pywt.frequency2scale(wavelet, normalised_frequencies)
    coeffs, frequencies = pywt.cwt(kappa, scales=scales, wavelet=wavelet, sampling_period=ds)
    energy = np.abs(coeffs) ** 2
    return energy, np.asarray(frequencies, dtype=float)


def _first_trough_toward_lower_frequency(column: np.ndarray, ridge_index: int) -> int:
    """Return first local minimum below a ridge, scanning toward lower frequencies."""
    col = np.asarray(column, dtype=float)
    ridge_index = int(np.clip(ridge_index, 0, col.size - 1))
    if ridge_index <= 1:
        return 0

    for idx in range(ridge_index - 1, 0, -1):
        if col[idx] <= col[idx - 1] and col[idx] <= col[idx + 1]:
            return int(idx)
    return 0


def compute_compound_segmentation_signal(
    curvature: np.ndarray,
    s: np.ndarray,
    *,
    width: float | None = None,
    meander_window_widths: float = 22.0,
    wavelet: str = "mexh",
    min_frequency_index: int = 1,
    max_frequency_index: int | None = None,
) -> CompoundSegmentationResult:
    """Compute the normalised CWT corridor-energy signal used for compound segmentation.

    The method follows the paper workflow: identify a local dominant CWT ridge,
    find the first trough toward lower pseudo-frequencies, integrate energy over
    the ridge-to-trough corridor, and normalise the resulting one-dimensional
    signal along the reach.
    """
    s = np.asarray(s, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    if s.shape != curvature.shape:
        raise ValueError("s and curvature must have the same shape.")

    energy, frequencies = _cwt_reach_energy(
        curvature,
        s,
        wavelet=wavelet,
        min_frequency_index=min_frequency_index,
        max_frequency_index=max_frequency_index,
    )

    ds = _median_spacing(s)
    if width is not None and float(width) > 0:
        window_length = float(meander_window_widths) * float(width)
        window_samples = _as_odd_window(round(window_length / ds), minimum=3)
    else:
        window_samples = _as_odd_window(round(0.05 * curvature.size), minimum=3)

    smoothed_energy = uniform_filter1d(energy, size=window_samples, axis=1, mode="nearest")
    ridge_indices = np.nanargmax(smoothed_energy, axis=0).astype(int)

    trough_indices = np.zeros_like(ridge_indices)
    corridor_energy = np.zeros(curvature.size, dtype=float)
    for j, ridge_idx in enumerate(ridge_indices):
        trough_idx = _first_trough_toward_lower_frequency(energy[:, j], int(ridge_idx))
        trough_indices[j] = trough_idx
        lo = min(trough_idx, ridge_idx)
        hi = max(trough_idx, ridge_idx)
        if hi == lo:
            corridor_energy[j] = float(energy[hi, j])
        else:
            corridor_energy[j] = float(trapezoid(energy[lo : hi + 1, j], frequencies[lo : hi + 1]))

    max_energy = float(np.nanmax(corridor_energy)) if corridor_energy.size else 0.0
    if not np.isfinite(max_energy) or max_energy <= 0:
        normalised = np.zeros_like(corridor_energy)
    else:
        normalised = np.clip(corridor_energy / max_energy, 0.0, 1.0)

    return CompoundSegmentationResult(
        energy=energy,
        frequencies=frequencies,
        s=s,
        ridge_indices=ridge_indices,
        trough_indices=trough_indices,
        corridor_energy=corridor_energy,
        normalised_energy=normalised,
        boundary_indices=np.asarray([], dtype=int),
    )


def _merge_short_units(boundaries: list[int], s: np.ndarray, min_unit_length: float) -> list[int]:
    """Merge neighbouring intervals shorter than ``min_unit_length``."""
    boundaries = sorted(set(int(b) for b in boundaries))
    if len(boundaries) <= 2:
        return boundaries

    changed = True
    while changed and len(boundaries) > 2:
        changed = False
        lengths = [float(s[b] - s[a]) for a, b in zip(boundaries[:-1], boundaries[1:])]
        short = [i for i, length in enumerate(lengths) if length < min_unit_length]
        if not short:
            break
        i = short[0]
        if i == 0:
            del boundaries[1]
        elif i == len(lengths) - 1:
            del boundaries[-2]
        else:
            left_length = float(s[boundaries[i + 1]] - s[boundaries[i - 1]])
            right_length = float(s[boundaries[i + 2]] - s[boundaries[i]])
            if left_length <= right_length:
                del boundaries[i]
            else:
                del boundaries[i + 1]
        changed = True
    return boundaries


def pick_compound_boundaries_from_signal(
    normalised_energy: np.ndarray,
    s: np.ndarray,
    *,
    valley_prominence: float = 0.05,
    min_unit_length: float | None = None,
) -> np.ndarray:
    """Pick compound-meander boundaries from local minima of the energy envelope."""
    signal = np.asarray(normalised_energy, dtype=float)
    s = np.asarray(s, dtype=float)
    if signal.shape != s.shape:
        raise ValueError("normalised_energy and s must have the same shape.")
    if signal.size < 3:
        return np.asarray([0, signal.size - 1], dtype=int)

    ds = _median_spacing(s)
    if min_unit_length is None:
        distance = 1
    else:
        distance = max(1, int(round(float(min_unit_length) / ds)))

    minima, _ = find_peaks(-signal, prominence=float(valley_prominence), distance=distance)
    boundaries = [0, *[int(i) for i in minima], signal.size - 1]

    if min_unit_length is not None:
        boundaries = _merge_short_units(boundaries, s, float(min_unit_length))

    return np.asarray(sorted(set(boundaries)), dtype=int)


def _compound_units_from_boundaries(boundary_indices: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.asarray(boundary_indices, dtype=int)
    if boundaries.size < 2:
        return []
    units = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if int(end) > int(start) + 2:
            units.append((int(start), int(end)))
    return units


def extract_compound_bends(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: float | np.ndarray | None = None,
    points_per_width: int = 25,
    unit_points: int = 201,
    meander_window_widths: float = 22.0,
    min_unit_widths: float = 8.0,
    valley_prominence: float = 0.05,
    wavelet: str = "mexh",
) -> tuple[list[CompoundBend], CompoundSegmentationResult]:
    """Extract simple/compound meander units from a centreline using CWT-energy valleys."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1 or x.size < 8:
        raise ValueError("x and y must be one-dimensional arrays with at least 8 points.")

    if width is None:
        mean_width = None
    elif np.ndim(width) == 0:
        mean_width = float(width)
    else:
        mean_width = float(np.nanmean(width))

    if mean_width and mean_width > 0:
        approx_length = cumulative_distance(x, y)[-1]
        resample_points = max(80, int(points_per_width * approx_length / mean_width))
    else:
        resample_points = max(200, len(x))

    s, xs, ys, curv = compute_centerline_curvature(x, y, resample_points=resample_points)
    width_resampled = None
    if np.ndim(width) == 1:
        width_resampled = _width_on_resampled_centerline(x, y, np.asarray(width, dtype=float), s)

    segmentation = compute_compound_segmentation_signal(
        curv,
        s,
        width=mean_width,
        meander_window_widths=meander_window_widths,
        wavelet=wavelet,
    )

    if mean_width and mean_width > 0:
        min_unit_length = float(min_unit_widths) * float(mean_width)
    else:
        min_unit_length = 0.05 * float(s[-1] - s[0])

    boundary_indices = pick_compound_boundaries_from_signal(
        segmentation.normalised_energy,
        s,
        valley_prominence=valley_prominence,
        min_unit_length=min_unit_length,
    )
    segmentation.boundary_indices = boundary_indices
    units = _compound_units_from_boundaries(boundary_indices)

    if not units:
        units = [(0, len(s) - 1)]
        segmentation.boundary_indices = np.asarray([0, len(s) - 1], dtype=int)

    inflections = detect_inflection_points(curv, s=s, min_spacing=None, include_endpoints=False)

    compounds: list[CompoundBend] = []
    for start, end in units:
        unit_width = mean_width
        if width_resampled is not None:
            unit_width = float(np.nanmean(width_resampled[start : end + 1]))

        raw_x = xs[start : end + 1]
        raw_y = ys[start : end + 1]
        raw_s = s[start : end + 1] - s[start]
        raw_c = curv[start : end + 1]

        norm_x, norm_y = normalise_bend(raw_x, raw_y, unit_width)
        norm_x, norm_y, norm_s = resample_polyline(norm_x, norm_y, n_points=unit_points)
        source_s = np.linspace(norm_s.min(), norm_s.max(), len(raw_c))
        norm_c = np.interp(norm_s, source_s, raw_c)

        chord = float(np.hypot(norm_x[-1] - norm_x[0], norm_y[-1] - norm_y[0]))
        length = float(s[end] - s[start])
        length_widths = None
        chord_widths = None
        if unit_width and unit_width > 0:
            length_widths = length / float(unit_width)
            chord_widths = float(np.hypot(raw_x[-1] - raw_x[0], raw_y[-1] - raw_y[0])) / float(unit_width)

        internal = inflections[(inflections > start) & (inflections < end)]
        n_internal = int(internal.size)
        n_lobes = max(1, n_internal + 1)

        compounds.append(
            CompoundBend(
                unit_id=len(compounds),
                start_index=int(start),
                end_index=int(end),
                x=norm_x,
                y=norm_y,
                s=norm_s,
                curvature=norm_c,
                raw_x=raw_x.copy(),
                raw_y=raw_y.copy(),
                raw_s=raw_s.copy(),
                width=unit_width,
                sinuosity=sinuosity(norm_x, norm_y),
                maturity_index=maturity_index(norm_x, norm_y),
                chord_length=chord,
                chord_widths=chord_widths,
                length=length,
                length_widths=length_widths,
                n_internal_inflections=n_internal,
                n_lobes=n_lobes,
                is_compound=bool(n_lobes >= 2),
            )
        )

    return compounds, segmentation


def compound_bends_to_metadata_rows(units: list[CompoundBend]) -> list[dict]:
    """Convert compound-bend objects to CSV-friendly dictionaries."""
    return [unit.metadata() for unit in units]
