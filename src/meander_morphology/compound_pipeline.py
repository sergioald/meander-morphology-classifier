from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .compound import compound_bends_to_metadata_rows, extract_compound_bends
from .cwt import save_spectrum_image, spectrum_image_from_geometry
from .io import read_centerline_table, write_bend_summary


def extract_compound_bends_and_spectra(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    width: float | None = None,
    width_column: str | None = "width",
    image_size: int = 64,
    points_per_width: int = 25,
    unit_points: int = 201,
    meander_window_widths: float = 22.0,
    min_unit_widths: float = 8.0,
    valley_prominence: float = 0.05,
) -> tuple[list, np.ndarray]:
    """Run centreline → compound units → CWT spectrum images.

    Outputs mirror the single-bend pipeline but use compound CWT-energy valley
    segmentation before producing one spectrum image per detected unit.
    """
    output_dir = Path(output_dir)
    spectra_dir = output_dir / "compound_spectra"
    diagnostics_dir = output_dir / "diagnostics"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    x, y, width_values = read_centerline_table(input_path, width_column=width_column)
    width_source = width_values if width_values is not None else width

    units, segmentation = extract_compound_bends(
        x,
        y,
        width=width_source,
        points_per_width=points_per_width,
        unit_points=unit_points,
        meander_window_widths=meander_window_widths,
        min_unit_widths=min_unit_widths,
        valley_prominence=valley_prominence,
    )

    spectra = []
    for unit in units:
        image = spectrum_image_from_geometry(
            unit.x,
            unit.y,
            image_size=image_size,
            target_points=unit_points,
        )
        spectra.append(image)
        save_spectrum_image(str(spectra_dir / f"compound_unit_{unit.unit_id:04d}.png"), image)

    spectra_array = np.asarray(spectra)
    write_bend_summary(output_dir / "compound_bend_summary.csv", compound_bends_to_metadata_rows(units))
    np.save(output_dir / "compound_spectra.npy", spectra_array)

    pd.DataFrame(
        {
            "s": segmentation.s,
            "normalised_corridor_energy": segmentation.normalised_energy,
            "corridor_energy": segmentation.corridor_energy,
            "ridge_index": segmentation.ridge_indices,
            "trough_index": segmentation.trough_indices,
            "is_boundary": np.isin(np.arange(segmentation.s.size), segmentation.boundary_indices),
        }
    ).to_csv(diagnostics_dir / "compound_segmentation_signal.csv", index=False)

    return units, spectra_array
