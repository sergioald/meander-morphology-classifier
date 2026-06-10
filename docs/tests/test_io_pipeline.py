from pathlib import Path

import numpy as np

from meander_morphology.io import read_centerline_table
from meander_morphology.pipeline import extract_bends_and_spectra


def test_read_centerline_table(tmp_path):
    path = tmp_path / "centerline.csv"
    path.write_text("x,y,width\n0,0,10\n1,1,10\n2,0,10\n")
    x, y, w = read_centerline_table(path)
    assert np.allclose(x, [0, 1, 2])
    assert w is not None


def test_extract_pipeline(tmp_path):
    x = np.linspace(0, 1000, 600)
    y = 80 * np.sin(2 * np.pi * x / 250)
    path = tmp_path / "centerline.csv"
    with path.open("w") as f:
        f.write("x,y,width\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi},{yi},40\n")
    bends, spectra = extract_bends_and_spectra(path, tmp_path / "out")
    assert len(bends) > 0
    assert spectra.shape[1:] == (64, 64)
    assert (tmp_path / "out" / "bend_summary.csv").exists()
