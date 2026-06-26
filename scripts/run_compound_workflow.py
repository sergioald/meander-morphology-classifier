#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from meander_morphology.compound_diagnostics import (
    save_diagnostic_latent_plot,
    save_latent_diagnostics,
)
from meander_morphology.compound_model import encode_compound_spectra, save_latent_outputs
from meander_morphology.compound_pipeline import extract_compound_bends_and_spectra


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the compound-bend workflow: centreline -> CWT-energy units -> "
            "spectrum images -> optional latent encoding and diagnostics."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV/TXT/DAT centreline file.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for the full workflow.")
    parser.add_argument("--width", type=float, default=None, help="Constant channel width if no width column is present.")
    parser.add_argument("--width-column", default="width", help="Width column name, or empty string to ignore.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--points-per-width", type=int, default=25)
    parser.add_argument("--unit-points", type=int, default=201)
    parser.add_argument("--meander-window-widths", type=float, default=22.0)
    parser.add_argument("--min-unit-widths", type=float, default=8.0)
    parser.add_argument("--valley-prominence", type=float, default=0.05)

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional encoder-only model or full compound autoencoder for latent encoding.",
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
        help="Optional world/background latent .npy cloud for diagnostics and plots.",
    )
    parser.add_argument("--plot", action="store_true", help="Save latent diagnostic plots when a model is supplied.")
    args = parser.parse_args()

    width_column = args.width_column or None
    args.output.mkdir(parents=True, exist_ok=True)

    units, spectra = extract_compound_bends_and_spectra(
        args.input,
        args.output,
        width=args.width,
        width_column=width_column,
        image_size=args.image_size,
        points_per_width=args.points_per_width,
        unit_points=args.unit_points,
        meander_window_widths=args.meander_window_widths,
        min_unit_widths=args.min_unit_widths,
        valley_prominence=args.valley_prominence,
    )

    n_compound = sum(unit.is_compound for unit in units)
    print(f"Extracted {len(units)} CWT-energy meander units")
    print(f"Compound/complex units: {n_compound}")
    print(f"Saved compound spectra with shape {spectra.shape} to {args.output}")

    if args.model is None:
        print("No --model supplied; skipped latent encoding.")
        return

    spectra_path = args.output / "compound_spectra.npy"
    summary_path = args.output / "compound_bend_summary.csv"
    encoded_dir = args.output / "encoded"

    latent = encode_compound_spectra(
        spectra_path,
        args.model,
        model_is_encoder=args.model_is_encoder,
        latent_layer_name=args.latent_layer_name,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
    )
    latent_result = save_latent_outputs(latent, encoded_dir, summary_path=summary_path)
    diagnostics_result = save_latent_diagnostics(
        latent_result.table,
        encoded_dir,
        background_latent_path=args.background_latent,
    )

    print(f"Encoded {latent.shape[0]} compound units into a {latent.shape[1]}-D latent space")
    print(f"Saved latent array: {latent_result.latent_path}")
    print(f"Saved latent table: {latent_result.table_path}")
    print(f"Saved latent diagnostics: {diagnostics_result.diagnostics_path}")
    print(f"Saved latent summary: {diagnostics_result.summary_path}")

    if args.plot:
        plot_path = save_diagnostic_latent_plot(
            diagnostics_result.table,
            encoded_dir / "compound_latent_diagnostics.png",
            background_latent_path=args.background_latent,
            colour_by="n_lobes",
            label_points=True,
        )
        print(f"Saved latent diagnostic plot: {plot_path}")


if __name__ == "__main__":
    main()
