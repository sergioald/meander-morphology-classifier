from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .bends import Bend


def plot_centerline_with_bends(x: np.ndarray, y: np.ndarray, bends: list[Bend], output: str | Path) -> None:
    """Save a centerline preview with bend indices."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, linewidth=1.5)
    for bend in bends:
        ax.text(bend.x[len(bend.x) // 2], bend.y[len(bend.y) // 2], str(bend.bend_id), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_latent_space(latent: np.ndarray, labels: np.ndarray | None, output: str | Path) -> None:
    """Save a latent-space scatter plot."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    if labels is None:
        ax.scatter(latent[:, 0], latent[:, 1], s=20)
    else:
        scatter = ax.scatter(latent[:, 0], latent[:, 1], c=labels, s=20)
        fig.colorbar(scatter, ax=ax, label="cluster")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
