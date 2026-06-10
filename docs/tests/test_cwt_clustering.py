import numpy as np

from meander_morphology.clustering import cluster_latent_space
from meander_morphology.cwt import spectrum_image


def test_spectrum_image_shape():
    signal = np.sin(np.linspace(0, 4 * np.pi, 201))
    image = spectrum_image(signal, image_size=64, n_scales=32)
    assert image.shape == (64, 64)
    assert image.min() >= 0
    assert image.max() <= 1


def test_cluster_latent_space():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(20, 2))
    result = cluster_latent_space(latent, n_clusters=3)
    assert result.labels.shape == (20,)
    assert result.centers.shape == (3, 2)
