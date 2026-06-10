from __future__ import annotations

from pathlib import Path

import numpy as np

from .bends import bends_to_metadata_rows, extract_single_bends
from .cwt import save_spectrum_image, spectrum_image
from .io import read_centerline_table, write_bend_summary


def extract_bends_and_spectra(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    width: float | None = None,
    width_column: str | None = "width",
    image_size: int = 64,
) -> tuple[list, np.ndarray]:
    """Run the centerline → bends → spectra workflow."""
    output_dir = Path(output_dir)
    spectra_dir = output_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    x, y, width_values = read_centerline_table(input_path, width_column=width_column)
    width_source = width_values if width_values is not None else width
    bends = extract_single_bends(x, y, width=width_source)

    spectra = []
    for bend in bends:
        image = spectrum_image(bend.curvature, image_size=image_size)
        spectra.append(image)
        save_spectrum_image(str(spectra_dir / f"bend_{bend.bend_id:04d}.png"), image)

    write_bend_summary(output_dir / "bend_summary.csv", bends_to_metadata_rows(bends))
    np.save(output_dir / "spectra.npy", np.asarray(spectra))
    return bends, np.asarray(spectra)
