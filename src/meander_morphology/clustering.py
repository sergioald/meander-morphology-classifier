from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(slots=True)
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def cluster_latent_space(
    latent: np.ndarray,
    *,
    n_clusters: int = 3,
    random_state: int = 35,
) -> ClusterResult:
    """Cluster latent coordinates with K-means."""
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2:
        raise ValueError("latent coordinates must have shape (n_samples, n_features).")
    if latent.shape[0] < n_clusters:
        raise ValueError("Number of samples must be at least the number of clusters.")
    model = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init="auto")
    labels = model.fit_predict(latent)
    return ClusterResult(labels=labels, centers=model.cluster_centers_, inertia=float(model.inertia_))


def label_names(labels: np.ndarray) -> list[str]:
    """Return user-facing cluster names for the single-bend workflow."""
    names = {
        0: "C1_or_model_cluster_0",
        1: "C2_or_model_cluster_1",
        2: "C3_or_model_cluster_2",
    }
    return [names.get(int(label), f"cluster_{int(label)}") for label in labels]
