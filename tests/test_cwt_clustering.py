from __future__ import annotations

import numpy as np

from meander_morphology.cwt import cwt_energy, spectrum_image, spectrum_image_from_geometry


def test_spectrum_image_shape():
    curvature = np.sin(np.linspace(0, np.pi, 201))
    image = spectrum_image(curvature, image_size=64)
    assert image.shape == (64, 64)
    assert np.isfinite(image).all()


def test_cwt_energy_shape():
    curvature = np.sin(np.linspace(0, np.pi, 201))
    result = cwt_energy(curvature)
    assert result.energy.shape == (200, 201)
    assert result.periods.shape == (200,)


def test_spectrum_image_from_geometry_shape():
    x = np.linspace(0, 4, 201)
    y = np.sin(np.linspace(0, np.pi, 201))
    image = spectrum_image_from_geometry(x, y, image_size=64, smooth=False)
    assert image.shape == (64, 64)
