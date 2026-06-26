import json

import numpy as np
import pandas as pd

from meander_morphology.compound_diagnostics import (
    build_latent_diagnostics,
    latent_summary_dict,
    save_diagnostic_latent_plot,
    save_latent_diagnostics,
)


def _example_latent_table():
    return pd.DataFrame(
        {
            "unit_id": [0, 1, 2],
            "n_lobes": [1, 2, 4],
            "is_compound": [False, True, True],
            "latent_1": [0.0, 3.0, -4.0],
            "latent_2": [0.0, 4.0, 3.0],
        }
    )


def test_build_latent_diagnostics_adds_interpretable_columns(tmp_path):
    background = np.array([[0.0, 0.0], [3.0, 4.0], [-4.0, 3.0], [10.0, 0.0]])
    background_path = tmp_path / "background.npy"
    np.save(background_path, background)

    result = build_latent_diagnostics(_example_latent_table(), background_latent_path=background_path)

    assert "latent_radius" in result.columns
    assert "latent_angle_deg" in result.columns
    assert "complexity_class" in result.columns
    assert "nearest_background_distance" in result.columns
    assert np.allclose(result["latent_radius"], [0.0, 5.0, 5.0])
    assert result.loc[0, "complexity_class"] == "simple"
    assert result.loc[1, "complexity_class"] == "compound_2_lobes"
    assert result.loc[2, "complexity_class"] == "compound_4plus_lobes"


def test_save_latent_diagnostics_writes_csv_and_json(tmp_path):
    output_dir = tmp_path / "diagnostics"

    saved = save_latent_diagnostics(_example_latent_table(), output_dir)

    assert saved.diagnostics_path.exists()
    assert saved.summary_path.exists()
    summary = json.loads(saved.summary_path.read_text(encoding="utf-8"))
    assert summary["n_units"] == 3
    assert summary["n_compound_units"] == 2
    assert summary["n_lobes_max"] == 4


def test_save_diagnostic_latent_plot_writes_png(tmp_path):
    output_path = tmp_path / "latent_plot.png"

    returned = save_diagnostic_latent_plot(_example_latent_table(), output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_latent_summary_dict_handles_basic_table():
    diagnostics = build_latent_diagnostics(_example_latent_table())
    summary = latent_summary_dict(diagnostics)

    assert summary["n_units"] == 3
    assert summary["latent_radius_max"] == 5.0
