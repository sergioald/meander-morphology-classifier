from __future__ import annotations

import numpy as np

from meander_morphology.cwt import legacy_compound_training_image_from_curvature


def test_legacy_compound_training_image_shape_and_range():
    s = np.linspace(0.0, 1.0, 201)
    curvature = np.sin(2 * np.pi * s) + 0.35 * np.sin(4 * np.pi * s)
    image = legacy_compound_training_image_from_curvature(curvature, image_size=64)
    assert image.shape == (64, 64)
    assert image.dtype == np.float32
    assert np.isfinite(image).all()
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    # Legacy training-style polarity should retain a bright background and darker structures.
    assert float(np.median(image)) > 0.5
