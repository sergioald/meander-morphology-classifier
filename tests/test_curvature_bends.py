import numpy as np

from meander_morphology.bends import extract_single_bends
from meander_morphology.curvature import compute_centerline_curvature, detect_inflection_points


def synthetic_centerline(n=600):
    x = np.linspace(0, 1000, n)
    y = 80 * np.sin(2 * np.pi * x / 250)
    width = np.full_like(x, 40.0)
    return x, y, width


def compound_like_centerline(n=800):
    x = np.linspace(0, 1000, n)
    y = 70 * np.sin(2 * np.pi * x / 250) + 20 * np.sin(2 * np.pi * x / 125)
    width = np.full_like(x, 40.0)
    return x, y, width


def test_curvature_and_inflections():
    x, y, _ = synthetic_centerline()
    s, xs, ys, c = compute_centerline_curvature(x, y, resample_points=500)
    pts = detect_inflection_points(c, s=s, include_endpoints=True)
    assert len(pts) >= 4
    assert len(s) == len(xs) == len(ys) == len(c)


def test_extract_single_bends():
    x, y, width = synthetic_centerline()
    bends = extract_single_bends(x, y, width=width, min_chord_widths=2.0)
    assert len(bends) >= 3
    assert bends[0].x.shape == (201,)
    assert bends[0].sinuosity >= 1.0
    assert bends[0].chord_widths is not None


def test_extraction_does_not_merge_by_inflection_spacing():
    x, y, width = compound_like_centerline()
    bends = extract_single_bends(x, y, width=width, min_chord_widths=0.0)
    boundaries = [(bend.start_index, bend.end_index) for bend in bends]
    assert len(boundaries) >= 4
    # Consecutive extracted bends should share endpoints. This confirms the
    # extractor is cutting at consecutive inflections instead of skipping
    # interior inflections and creating compound-looking units.
    for previous, current in zip(bends[:-1], bends[1:]):
        assert previous.end_index == current.start_index
