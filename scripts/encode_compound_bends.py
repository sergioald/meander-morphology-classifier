#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from meander_morphology.compound_model import (
    encode_compound_spectra,
    save_latent_outputs,
    save_latent_plot,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode compound-bend CWT spectra into latent-space coordinates."
    )
    parser.add_argument(
        "--spectra",
        required=True,
        type=Path,
        help="Path to compound_spectra.npy from scripts/extract_compound_bends.py.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to encoder-only model or full compound autoencoder.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where latent CSV/NPY outputs will be written.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional compound_bend_summary.csv to merge with latent coordinates.",
    )
    parser.add_argument(
        "--model-is-encoder",
        action="store_true",
        help="Set when --model already points to encoder_only.keras or encoder_only.h5.",
    )
    parser.add_argument(
        "--latent-layer-name",
        default="Latent_Space",
        help="Latent layer name used when --model is a full autoencoder.",
    )
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--background-latent",
        type=Path,
        default=None,
        help="Optional world/background latent .npy cloud for the diagnostic plot.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a simple 2-D latent-space PNG diagnostic plot.",
    )
    args = parser.parse_args()

    latent = encode_compound_spectra(
        args.spectra,
        args.model,
        model_is_encoder=args.model_is_encoder,
        latent_layer_name=args.latent_layer_name,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
    )
    result = save_latent_outputs(latent, args.output, summary_path=args.summary)

    print(f"Encoded {latent.shape[0]} compound units into a {latent.shape[1]}-D latent space")
    print(f"Saved latent array: {result.latent_path}")
    print(f"Saved latent table: {result.table_path}")

    if args.plot:
        plot_path = save_latent_plot(
            latent,
            args.output / "compound_latent_space.png",
            background_latent_path=args.background_latent,
        )
        print(f"Saved latent plot: {plot_path}")


if __name__ == "__main__":
    main()
