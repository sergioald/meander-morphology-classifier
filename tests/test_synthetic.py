from __future__ import annotations

import numpy as np

from meander_morphology.cwt import spectrum_image
from meander_morphology.synthetic import (
    KinoshitaParameters,
    generate_kinoshita_bends,
    kinoshita_centerline,
    sample_kinoshita_parameters,
)


def test_kinoshita_centerline_shapes_and_finite_values():
    params = KinoshitaParameters(wavelength=20.0, theta_1=1.2, theta_3r=0.2, theta_3i=-0.1)
    x, y, s, theta, curvature = kinoshita_centerline(params, n_points=101)
    assert x.shape == y.shape == s.shape == theta.shape == curvature.shape == (101,)
    assert np.isfinite(x).all()
    assert np.isfinite(curvature).all()


def test_sample_kinoshita_parameters_is_reproducible():
    p1 = sample_kinoshita_parameters(3, seed=35)
    p2 = sample_kinoshita_parameters(3, seed=35)
    assert [p.asdict() for p in p1] == [p.asdict() for p in p2]


def test_generate_kinoshita_bends_and_spectrum_image():
    bends = generate_kinoshita_bends(5, seed=35, source_points=301, bend_points=101)
    assert len(bends) == 5
    assert bends[0].x.shape == (101,)
    assert bends[0].curvature.shape == (101,)
    image = spectrum_image(bends[0].curvature, image_size=32, n_scales=40)
    assert image.shape == (32, 32)
    assert np.isfinite(image).all()
