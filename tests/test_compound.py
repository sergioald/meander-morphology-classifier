import numpy as np
import pandas as pd

from meander_morphology.compound import (
    extract_compound_bends,
    pick_compound_boundaries_from_signal,
)
from meander_morphology.compound_pipeline import extract_compound_bends_and_spectra


def test_pick_compound_boundaries_from_signal_detects_valleys():
    s = np.linspace(0.0, 100.0, 501)
    signal = (
        0.15
        + np.exp(-((s - 20.0) / 8.0) ** 2)
        + np.exp(-((s - 55.0) / 8.0) ** 2)
        + np.exp(-((s - 85.0) / 8.0) ** 2)
    )
    signal = signal / signal.max()

    boundaries = pick_compound_boundaries_from_signal(
        signal,
        s,
        valley_prominence=0.05,
        min_unit_length=15.0,
    )

    assert boundaries[0] == 0
    assert boundaries[-1] == len(s) - 1
    assert len(boundaries) >= 3


def test_extract_compound_bends_returns_metadata():
    t = np.linspace(0.0, 8.0 * np.pi, 800)
    x = 5.0 * t
    y = 10.0 * np.sin(t) + 2.5 * np.sin(3.0 * t + 0.4)

    units, segmentation = extract_compound_bends(
        x,
        y,
        width=5.0,
        points_per_width=8,
        meander_window_widths=18.0,
        min_unit_widths=6.0,
        valley_prominence=0.02,
    )

    assert len(units) >= 1
    assert segmentation.normalised_energy.shape == segmentation.s.shape
    assert segmentation.boundary_indices[0] == 0
    assert segmentation.boundary_indices[-1] == len(segmentation.s) - 1
    row = units[0].metadata()
    assert "n_lobes" in row
    assert "is_compound" in row


def test_compound_pipeline_writes_outputs(tmp_path):
    t = np.linspace(0.0, 6.0 * np.pi, 500)
    df = pd.DataFrame(
        {
            "x": 4.0 * t,
            "y": 8.0 * np.sin(t) + 2.0 * np.sin(3.0 * t),
            "width": np.full_like(t, 4.0),
        }
    )
    input_file = tmp_path / "centreline.csv"
    output_dir = tmp_path / "compound_output"
    df.to_csv(input_file, index=False)

    units, spectra = extract_compound_bends_and_spectra(
        input_file,
        output_dir,
        image_size=32,
        points_per_width=6,
        meander_window_widths=18.0,
        min_unit_widths=6.0,
        valley_prominence=0.02,
    )

    assert len(units) == spectra.shape[0]
    assert spectra.shape[1:] == (32, 32)
    assert (output_dir / "compound_bend_summary.csv").exists()
    assert (output_dir / "compound_spectra.npy").exists()
    assert (output_dir / "diagnostics" / "compound_segmentation_signal.csv").exists()
