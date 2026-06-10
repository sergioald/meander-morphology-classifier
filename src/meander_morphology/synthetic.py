from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median

import numpy as np
from scipy.stats import gamma

from .curvature import detect_inflection_points
from .geometry import maturity_index, normalise_bend, resample_polyline, sinuosity


@dataclass(slots=True)
class KinoshitaParameters:
    """Parameters for one dimensionless Kinoshita meander centerline."""

    wavelength: float
    theta_1: float
    theta_3r: float
    theta_3i: float

    def asdict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class SyntheticBend:
    """One normalized synthetic single bend and its source parameters."""

    bend_id: int
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    curvature: np.ndarray
    theta: np.ndarray
    parameters: KinoshitaParameters
    source_start_index: int
    source_end_index: int
    sinuosity: float
    maturity_index: float

    def metadata(self) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "bend_id": self.bend_id,
            "source_start_index": self.source_start_index,
            "source_end_index": self.source_end_index,
            "sinuosity": self.sinuosity,
            "maturity_index": self.maturity_index,
        }
        row.update(self.parameters.asdict())
        return row


def sample_kinoshita_parameters(
    n_samples: int,
    *,
    seed: int = 35,
    wavelength_gamma_shape: float = 12.727632868227522,
    wavelength_gamma_scale: float = 0.026491678981197,
) -> list[KinoshitaParameters]:
    """Sample Kinoshita parameters following the legacy research scripts.

    The original generation scripts used a gamma distribution for wavenumber
    estimates and uniform distributions for the Kinoshita amplitude, fattening
    and skewing terms. This helper keeps those defaults but returns independent
    random samples instead of creating the full Cartesian product, which makes it
    safe for examples and tests.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    theta_1_low = 4 / np.pi - 1 / 3
    theta_1_high = 4 / np.pi + 1 / 3

    wavenumbers = gamma.rvs(
        wavelength_gamma_shape,
        loc=0,
        scale=wavelength_gamma_scale,
        size=n_samples,
        random_state=rng,
    )
    wavelengths = 2.0 * np.pi / wavenumbers
    theta_1 = rng.uniform(theta_1_low, theta_1_high, n_samples)
    theta_3r = rng.uniform(-1.0, 1.0, n_samples)
    theta_3i = rng.uniform(-1.0, 1.0, n_samples)

    return [
        KinoshitaParameters(
            wavelength=float(wavelengths[i]),
            theta_1=float(theta_1[i]),
            theta_3r=float(theta_3r[i]),
            theta_3i=float(theta_3i[i]),
        )
        for i in range(n_samples)
    ]


def legacy_parameter_grid(
    *,
    n_parameter_values: int = 77,
    seed: int = 35,
    use_median_wavelength: bool = True,
) -> dict[str, np.ndarray]:
    """Return the parameter arrays used by the original full-grid generator.

    The full grid has ``n_parameter_values**3`` bends when a single median
    wavelength is used. With the default of 77 values this is 456,533 parameter
    combinations, so this function returns only the arrays and does not generate
    all bends automatically.
    """
    if n_parameter_values <= 1:
        raise ValueError("n_parameter_values must be greater than one")

    rng = np.random.default_rng(seed)
    wavenumber_temp = gamma.rvs(
        12.727632868227522,
        loc=0,
        scale=0.026491678981197,
        size=50,
        random_state=rng,
    )
    wavelength_values = 2.0 * np.pi / wavenumber_temp
    if use_median_wavelength:
        wavelength_values = np.asarray([2.0 * np.pi / median(wavenumber_temp)], dtype=float)

    theta_1_low = 4 / np.pi - 1 / 3
    theta_1_high = 4 / np.pi + 1 / 3
    return {
        "wavelength": np.asarray(wavelength_values, dtype=float),
        "theta_1": rng.uniform(theta_1_low, theta_1_high, n_parameter_values),
        "theta_3r": rng.uniform(-1.0, 1.0, n_parameter_values),
        "theta_3i": rng.uniform(-1.0, 1.0, n_parameter_values),
    }


def kinoshita_centerline(
    parameters: KinoshitaParameters,
    *,
    n_points: int = 801,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a dimensionless Kinoshita centerline.

    Returns ``x, y, s, theta, curvature``. Curvature is analytic with respect to
    arclength, matching the legacy implementation.
    """
    if n_points < 20:
        raise ValueError("n_points must be at least 20")
    wavelength = float(parameters.wavelength)
    s = np.linspace(0.0, wavelength, int(n_points))
    lam = 2.0 * np.pi / wavelength
    theta = (
        parameters.theta_1 * np.sin(lam * s)
        + parameters.theta_3r * np.cos(3.0 * lam * s)
        + parameters.theta_3i * np.sin(3.0 * lam * s)
    )
    curvature = (
        lam * parameters.theta_1 * np.cos(lam * s)
        - 3.0 * lam * parameters.theta_3r * np.sin(3.0 * lam * s)
        + 3.0 * lam * parameters.theta_3i * np.cos(3.0 * lam * s)
    )

    ds = np.diff(s)
    x = np.zeros_like(s)
    y = np.zeros_like(s)
    x[1:] = np.cumsum(np.cos(theta[:-1]) * ds)
    y[1:] = np.cumsum(np.sin(theta[:-1]) * ds)
    return x, y, s, theta, curvature


def extract_single_bend_from_kinoshita(
    parameters: KinoshitaParameters,
    *,
    source_points: int = 801,
    bend_points: int = 201,
    interval: str = "middle",
) -> SyntheticBend:
    """Generate one normalized single bend from a Kinoshita centerline."""
    x, y, s, theta, curvature = kinoshita_centerline(parameters, n_points=source_points)
    inflections = detect_inflection_points(curvature, include_endpoints=False)
    if len(inflections) < 2:
        raise ValueError("Generated centerline has fewer than two interior inflection points")

    starts = inflections[:-1]
    ends = inflections[1:]
    if interval == "middle":
        idx = len(starts) // 2
    elif interval == "longest":
        idx = int(np.argmax(s[ends] - s[starts]))
    else:
        raise ValueError("interval must be 'middle' or 'longest'")

    start = int(starts[idx])
    end = int(ends[idx])
    seg_x = x[start : end + 1]
    seg_y = y[start : end + 1]
    seg_s = s[start : end + 1]
    seg_c = curvature[start : end + 1]
    seg_t = theta[start : end + 1]

    bend_x, bend_y = normalise_bend(seg_x, seg_y, width=None)
    bend_x, bend_y, bend_s = resample_polyline(bend_x, bend_y, n_points=bend_points)
    source_s = np.linspace(float(bend_s.min()), float(bend_s.max()), len(seg_c))
    bend_c = np.interp(bend_s, source_s, seg_c)
    bend_t = np.interp(bend_s, source_s, seg_t)

    return SyntheticBend(
        bend_id=0,
        x=bend_x,
        y=bend_y,
        s=bend_s,
        curvature=bend_c,
        theta=bend_t,
        parameters=parameters,
        source_start_index=start,
        source_end_index=end,
        sinuosity=sinuosity(bend_x, bend_y),
        maturity_index=maturity_index(bend_x, bend_y),
    )


def generate_kinoshita_bends(
    n_bends: int,
    *,
    seed: int = 35,
    source_points: int = 801,
    bend_points: int = 201,
    max_attempts_factor: int = 5,
) -> list[SyntheticBend]:
    """Generate a list of normalized synthetic Kinoshita single bends."""
    if n_bends <= 0:
        raise ValueError("n_bends must be positive")
    max_attempts = max(n_bends, n_bends * max_attempts_factor)
    parameters = sample_kinoshita_parameters(max_attempts, seed=seed)

    bends: list[SyntheticBend] = []
    for params in parameters:
        try:
            bend = extract_single_bend_from_kinoshita(
                params,
                source_points=source_points,
                bend_points=bend_points,
            )
        except ValueError:
            continue
        bend.bend_id = len(bends)
        bends.append(bend)
        if len(bends) >= n_bends:
            break

    if len(bends) < n_bends:
        raise RuntimeError(f"Only generated {len(bends)} valid bends out of {n_bends} requested")
    return bends


def bends_to_arrays(bends: list[SyntheticBend]) -> dict[str, np.ndarray]:
    """Convert synthetic bend objects into stacked arrays."""
    return {
        "x": np.asarray([bend.x for bend in bends], dtype=float),
        "y": np.asarray([bend.y for bend in bends], dtype=float),
        "s": np.asarray([bend.s for bend in bends], dtype=float),
        "curvature": np.asarray([bend.curvature for bend in bends], dtype=float),
        "theta": np.asarray([bend.theta for bend in bends], dtype=float),
    }
