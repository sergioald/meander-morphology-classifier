from __future__ import annotations

from pathlib import Path

import numpy as np

from .bends import bends_to_metadata_rows, extract_single_bends
from .cwt import save_spectrum_image, spectrum_image_from_geometry
from .io import read_centerline_table, write_bend_summary


def extract_bends_and_spectra(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    width: float | None = None,
    width_column: str | None = "width",
    image_size: int = 64,
    min_chord_widths: float | None = None,
    include_edge_bends: bool = False,
    endpoint_mode: str = "auto",
    endpoint_curvature_tolerance: float = 0.10,
    cwt_pad: bool | None = None,
    cwt_pad_fraction: float | None = None,
    cwt_max_scale_fraction: float | None = None,
) -> tuple[list, np.ndarray]:
    """Run centerline → single bends → legacy-compatible isolated CWT spectra.

    The CWT spectrum is computed from each selected bend's normalized isolated
    geometry. The bend is reflected only while recomputing curvature and is
    cropped before the CWT, matching the original research scripts.
    """
    del cwt_pad, cwt_pad_fraction, cwt_max_scale_fraction  # kept for backward-compatible calls

    output_dir = Path(output_dir)
    spectra_dir = output_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    x, y, width_values = read_centerline_table(input_path, width_column=width_column)
    width_source = width_values if width_values is not None else width
    bends = extract_single_bends(
        x,
        y,
        width=width_source,
        min_chord_widths=min_chord_widths,
        include_edge_bends=include_edge_bends,
        endpoint_mode=endpoint_mode,
        endpoint_curvature_tolerance=endpoint_curvature_tolerance,
    )

    spectra = []
    for bend in bends:
        image = spectrum_image_from_geometry(
            bend.x,
            bend.y,
            image_size=image_size,
            target_points=201,
        )
        spectra.append(image)
        save_spectrum_image(str(spectra_dir / f"bend_{bend.bend_id:04d}.png"), image)

    write_bend_summary(output_dir / "bend_summary.csv", bends_to_metadata_rows(bends))
    np.save(output_dir / "spectra.npy", np.asarray(spectra))
    return bends, np.asarray(spectra)
