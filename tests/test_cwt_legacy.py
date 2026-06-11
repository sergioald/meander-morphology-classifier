from __future__ import annotations

import numpy as np

from meander_morphology.cwt import (
    cwt_energy,
    cwt_energy_from_geometry,
    legacy_single_bend_curvature,
    spectrum_image_from_geometry,
)


def test_cwt_energy_uses_cropped_curvature_width():
    curvature = np.sin(np.linspace(0, np.pi, 201))
    result = cwt_energy(curvature)
    assert result.energy.shape == (200, 201)
    assert result.s.shape == (201,)
    assert result.periods.shape == (200,)


def test_legacy_curvature_reflects_then_crops_to_center():
    x = np.linspace(0, 4, 201)
    y = np.sin(np.linspace(0, np.pi, 201))
    s, xc, yc, curvature = legacy_single_bend_curvature(x, y, target_points=201, smooth=False)
    assert s.shape == (201,)
    assert xc.shape == (201,)
    assert yc.shape == (201,)
    assert curvature.shape == (201,)
    assert np.isfinite(curvature).all()


def test_spectrum_image_from_geometry_shape():
    x = np.linspace(0, 4, 201)
    y = np.sin(np.linspace(0, np.pi, 201))
    image = spectrum_image_from_geometry(x, y, image_size=64, smooth=False)
    assert image.shape == (64, 64)
    assert 0.0 <= image.min() <= image.max() <= 1.0


def test_cwt_energy_from_geometry_shape():
    x = np.linspace(0, 4, 201)
    y = np.sin(np.linspace(0, np.pi, 201))
    result = cwt_energy_from_geometry(x, y, target_points=201, smooth=False)
    assert result.energy.shape == (200, 201)
