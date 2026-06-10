#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meander_morphology.synthetic import legacy_parameter_grid, sample_kinoshita_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Kinoshita parameter distributions.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--legacy-grid",
        action="store_true",
        help="Plot the legacy full-grid parameter arrays instead of random samples",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.legacy_grid:
        grid = legacy_parameter_grid(n_parameter_values=77, seed=args.seed)
        data = pd.DataFrame(
            {
                "wavelength": np.repeat(grid["wavelength"], len(grid["theta_1"])),
                "theta_1": grid["theta_1"],
                "theta_3r": grid["theta_3r"],
                "theta_3i": grid["theta_3i"],
            }
        )
    else:
        params = sample_kinoshita_parameters(args.n_samples, seed=args.seed)
        data = pd.DataFrame([param.asdict() for param in params])

    data.to_csv(args.output / "kinoshita_parameters.csv", index=False)

    labels = ["wavelength", "theta_1", "theta_3r", "theta_3i"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, column in zip(axes, labels):
        ax.hist(data[column], bins=args.bins, density=True, edgecolor="black")
        ax.set_xlabel(column)
        ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig(args.output / "kinoshita_parameter_histograms.png", dpi=300)
    plt.close(fig)
    print(f"Saved parameter table and histogram to {args.output}")


if __name__ == "__main__":
    main()
