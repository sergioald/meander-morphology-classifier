import numpy as np

from meander_morphology.geometry import cumulative_distance, normalise_bend, resample_polyline, sinuosity


def test_cumulative_distance():
    s = cumulative_distance(np.array([0, 3]), np.array([0, 4]))
    assert np.allclose(s, [0, 5])


def test_resample_polyline():
    x, y, s = resample_polyline(np.array([0, 10]), np.array([0, 0]), n_points=6)
    assert len(x) == 6
    assert np.allclose(y, 0)
    assert np.isclose(s[-1], 10)


def test_normalise_bend_orients_positive_y():
    x, y = normalise_bend(np.array([0, 1, 2]), np.array([0, -1, 0]), width=1)
    assert np.isclose(x[0], 0)
    assert y.max() > 0


def test_sinuosity_line_is_one():
    assert np.isclose(sinuosity(np.array([0, 1, 2]), np.array([0, 0, 0])), 1.0)
