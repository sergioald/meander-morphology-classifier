import numpy as np

from meander_morphology.cwt import (
    legacy_compound_training_image_from_curvature,
    legacy_compound_training_preview_from_curvature,
)


def test_legacy_compound_training_preview_shapes():
    s = np.linspace(0.0, 1.0, 251)
    curvature = 0.25 * np.sin(4 * np.pi * s) + 0.1 * np.sin(9 * np.pi * s)
    out = legacy_compound_training_preview_from_curvature(curvature, image_size=64)

    assert {"cwt_matrix", "s_axis", "l_axis", "model_image"} <= set(out)
    assert out["model_image"].shape == (64, 64)
    assert out["cwt_matrix"].ndim == 2
    assert out["cwt_matrix"].shape[1] == out["s_axis"].shape[0]
    assert out["cwt_matrix"].shape[0] == out["l_axis"].shape[0]
    assert np.nanmin(out["s_axis"]) >= 0.0
    assert np.nanmax(out["s_axis"]) <= 1.0
    assert np.all(np.isfinite(out["model_image"]))


def test_legacy_compound_model_image_is_rasterised_training_plot():
    s = np.linspace(0.0, 1.0, 251)
    curvature = 0.25 * np.sin(4 * np.pi * s) + 0.1 * np.sin(9 * np.pi * s)
    image = legacy_compound_training_image_from_curvature(curvature, image_size=64)
    assert image.shape == (64, 64)
    assert image.min() >= 0.0
    assert image.max() <= 1.0
    assert image.std() > 0.001
