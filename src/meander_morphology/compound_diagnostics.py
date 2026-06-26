from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CompoundDiagnosticsResult:
    """Paths and in-memory table produced by the compound latent diagnostics step."""

    table: pd.DataFrame
    diagnostics_path: Path
    summary_path: Path


def _read_table(table_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(table_or_path, pd.DataFrame):
        return table_or_path.copy()
    return pd.read_csv(Path(table_or_path))


def _latent_columns(table: pd.DataFrame, latent_columns: tuple[str, str] | None = None) -> tuple[str, str]:
    if latent_columns is not None:
        col1, col2 = latent_columns
    else:
        col1, col2 = "latent_1", "latent_2"
    missing = [column for column in (col1, col2) if column not in table.columns]
    if missing:
        raise ValueError(f"Missing latent coordinate columns: {missing}")
    return col1, col2


def _load_background(background_latent_path: str | Path | None) -> np.ndarray | None:
    if background_latent_path is None:
        return None
    background = np.asarray(np.load(Path(background_latent_path)), dtype=float)
    if background.ndim != 2 or background.shape[1] < 2:
        raise ValueError("background latent cloud must have shape (N, >=2).")
    return background[:, :2]


def _nearest_background_distance(points: np.ndarray, background: np.ndarray, *, chunk_size: int = 50_000) -> np.ndarray:
    """Return nearest Euclidean distance to a background cloud using chunks.

    The world latent cloud can be large, so the computation avoids creating a
    full ``N_units x N_background`` distance matrix when it is not needed.
    """
    points = np.asarray(points, dtype=float)
    background = np.asarray(background, dtype=float)
    nearest = np.full(points.shape[0], np.inf, dtype=float)
    for start in range(0, background.shape[0], int(chunk_size)):
        chunk = background[start : start + int(chunk_size)]
        diff = points[:, None, :] - chunk[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        nearest = np.minimum(nearest, np.nanmin(dist, axis=1))
    return nearest


def _complexity_label(n_lobes: object) -> str:
    try:
        value = int(n_lobes)
    except (TypeError, ValueError):
        return "unknown"
    if value <= 1:
        return "simple"
    if value == 2:
        return "compound_2_lobes"
    if value == 3:
        return "compound_3_lobes"
    return "compound_4plus_lobes"


def build_latent_diagnostics(
    table_or_path: pd.DataFrame | str | Path,
    *,
    background_latent_path: str | Path | None = None,
    latent_columns: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Add interpretable diagnostic columns to a compound latent-coordinate table.

    The input is usually ``compound_latent.csv`` produced by
    ``scripts/encode_compound_bends.py``. The returned table keeps all original
    metadata and appends geometric latent-space metrics:

    - radial distance from the latent origin;
    - polar angle in degrees;
    - optional distance to the background latent-cloud centroid;
    - optional nearest-neighbour distance to the background cloud;
    - optional percentile of the unit radius relative to the background cloud.
    """
    table = _read_table(table_or_path)
    col1, col2 = _latent_columns(table, latent_columns)

    points = table[[col1, col2]].to_numpy(dtype=float)
    radius = np.sqrt(np.sum(points**2, axis=1))
    angle_deg = np.degrees(np.arctan2(points[:, 1], points[:, 0]))

    table["latent_radius"] = radius
    table["latent_angle_deg"] = angle_deg

    if "n_lobes" in table.columns:
        table["complexity_class"] = table["n_lobes"].map(_complexity_label)

    background = _load_background(background_latent_path)
    if background is not None and background.size:
        centroid = np.nanmean(background, axis=0)
        background_radius = np.sqrt(np.sum(background**2, axis=1))
        background_radius = background_radius[np.isfinite(background_radius)]
        table["background_centroid_distance"] = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        table["nearest_background_distance"] = _nearest_background_distance(points, background)
        if background_radius.size:
            table["background_radius_percentile"] = [
                float(100.0 * np.mean(background_radius <= value)) for value in radius
            ]

    return table


def latent_summary_dict(table: pd.DataFrame) -> dict:
    """Return a compact JSON-serialisable summary of a diagnostic latent table."""
    summary: dict[str, object] = {"n_units": int(len(table))}
    if "is_compound" in table.columns:
        summary["n_compound_units"] = int(np.asarray(table["is_compound"], dtype=bool).sum())
    if "n_lobes" in table.columns and len(table):
        summary["n_lobes_min"] = int(np.nanmin(table["n_lobes"]))
        summary["n_lobes_max"] = int(np.nanmax(table["n_lobes"]))
        summary["n_lobes_mean"] = float(np.nanmean(table["n_lobes"]))
    for column in [
        "latent_1",
        "latent_2",
        "latent_radius",
        "background_centroid_distance",
        "nearest_background_distance",
        "background_radius_percentile",
    ]:
        if column in table.columns and len(table):
            values = np.asarray(table[column], dtype=float)
            summary[f"{column}_min"] = float(np.nanmin(values))
            summary[f"{column}_max"] = float(np.nanmax(values))
            summary[f"{column}_mean"] = float(np.nanmean(values))
    return summary


def save_latent_diagnostics(
    table_or_path: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    background_latent_path: str | Path | None = None,
    prefix: str = "compound",
) -> CompoundDiagnosticsResult:
    """Save an enriched latent table and a compact JSON summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_latent_diagnostics(table_or_path, background_latent_path=background_latent_path)
    diagnostics_path = output_dir / f"{prefix}_latent_diagnostics.csv"
    summary_path = output_dir / f"{prefix}_latent_summary.json"

    table.to_csv(diagnostics_path, index=False)
    summary_path.write_text(json.dumps(latent_summary_dict(table), indent=2), encoding="utf-8")

    return CompoundDiagnosticsResult(
        table=table,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
    )


def save_diagnostic_latent_plot(
    table_or_path: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    background_latent_path: str | Path | None = None,
    colour_by: str = "n_lobes",
    label_points: bool = True,
    title: str = "Compound-bend latent diagnostics",
) -> Path:
    """Save a labelled latent-space diagnostic plot from a latent CSV/table."""
    import matplotlib.pyplot as plt

    table = build_latent_diagnostics(table_or_path, background_latent_path=background_latent_path)
    col1, col2 = _latent_columns(table)
    points = table[[col1, col2]].to_numpy(dtype=float)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    background = _load_background(background_latent_path)
    if background is not None:
        ax.scatter(background[:, 0], background[:, 1], s=2, alpha=0.12, label="background")

    if colour_by in table.columns:
        values = table[colour_by].to_numpy()
        if np.issubdtype(values.dtype, np.number):
            scatter = ax.scatter(points[:, 0], points[:, 1], s=45, c=values, label="compound units")
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(colour_by)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=45, label="compound units")
    else:
        ax.scatter(points[:, 0], points[:, 1], s=45, label="compound units")

    if label_points:
        label_source = table["unit_id"] if "unit_id" in table.columns else range(len(table))
        for label, (x, y) in zip(label_source, points):
            ax.annotate(str(label), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
