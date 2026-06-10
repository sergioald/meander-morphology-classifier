#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from meander_morphology.cwt import save_spectrum_image, spectrum_image
from meander_morphology.synthetic import bends_to_arrays, generate_kinoshita_bends


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic single bends from Kinoshita centerlines."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--n-bends", type=int, default=100, help="Number of bends to generate")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--source-points", type=int, default=801)
    parser.add_argument("--bend-points", type=int, default=201)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save each spectrum as a PNG image in output/spectra_png",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    bends = generate_kinoshita_bends(
        args.n_bends,
        seed=args.seed,
        source_points=args.source_points,
        bend_points=args.bend_points,
    )
    arrays = bends_to_arrays(bends)
    for name, array in arrays.items():
        np.save(args.output / f"{name}_bends.npy", array)

    spectra = np.asarray(
        [spectrum_image(curvature, image_size=args.image_size) for curvature in arrays["curvature"]],
        dtype=float,
    )
    np.save(args.output / "spectra.npy", spectra)

    if args.save_png:
        png_dir = args.output / "spectra_png"
        png_dir.mkdir(parents=True, exist_ok=True)
        for bend, image in zip(bends, spectra):
            save_spectrum_image(str(png_dir / f"synthetic_bend_{bend.bend_id:04d}.png"), image)

    pd.DataFrame([bend.metadata() for bend in bends]).to_csv(
        args.output / "synthetic_bend_metadata.csv",
        index=False,
    )
    print(f"Generated {len(bends)} synthetic bends in {args.output}")


if __name__ == "__main__":
    main()
