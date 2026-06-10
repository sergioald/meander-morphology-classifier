import numpy as np

from meander_morphology.clustering import cluster_latent_space
from meander_morphology.cwt import cwt_energy, mirror_pad_signal, spectrum_image


def test_spectrum_image_shape():
    curvature = np.sin(np.linspace(0, 2 * np.pi, 201))
    image = spectrum_image(curvature, image_size=64)
    assert image.shape == (64, 64)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_mirror_padded_cwt_is_cropped_to_original_length():
    curvature = np.sin(np.linspace(0, 2 * np.pi, 201))
    padded, crop = mirror_pad_signal(curvature)
    assert len(padded) > len(curvature)
    assert len(padded[crop]) == len(curvature)
    energy, scales = cwt_energy(curvature, n_scales=32, pad=True)
    assert energy.shape == (32, len(curvature))
    assert scales.shape == (32,)


def test_cluster_latent_space():
    rng = np.random.default_rng(35)
    latent = rng.normal(size=(12, 2))
    result = cluster_latent_space(latent, n_clusters=3)
    assert result.labels.shape == (12,)
    assert result.centers.shape == (3, 2)
