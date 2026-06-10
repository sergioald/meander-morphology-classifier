from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_centerline_table(
    path: str | Path,
    *,
    x_column: str = "x",
    y_column: str = "y",
    width_column: str | None = "width",
    delimiter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read centerline coordinates from CSV/TXT/DAT data.

    The preferred format has named columns ``x``, ``y`` and optionally ``width``.
    Files without headers are interpreted as at least two numeric columns, with the
    fourth column used as width when present, matching the legacy research files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, sep=delimiter, engine="python" if delimiter is None else None)
        if x_column in df.columns and y_column in df.columns:
            x = df[x_column].to_numpy(dtype=float)
            y = df[y_column].to_numpy(dtype=float)
            width = None
            if width_column and width_column in df.columns:
                width = df[width_column].to_numpy(dtype=float)
            return x, y, width
    except Exception:
        pass

    arr = np.loadtxt(path, delimiter=delimiter)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Centerline file must contain at least x and y columns.")
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    width = arr[:, 3].astype(float) if arr.shape[1] >= 4 else None
    return x, y, width


def write_bend_summary(path: str | Path, rows: list[dict]) -> None:
    """Write bend metadata to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
