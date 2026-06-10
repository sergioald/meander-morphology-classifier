#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from meander_morphology.pipeline import extract_bends_and_spectra


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract single bends and CWT spectra from a centerline file.")
    parser.add_argument("--input", required=True, type=Path, help="CSV/TXT/DAT centerline file.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--width", type=float, default=None, help="Constant channel width if no width column is present.")
    parser.add_argument("--width-column", default="width", help="Width column name, or empty string to ignore.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--min-chord-widths",
        type=float,
        default=0.0,
        help="Minimum bend chord length in channel widths. Use 0 to disable.",
    )
    parser.add_argument(
        "--include-edge-bends",
        action="store_true",
        help="Include partial first/last bends bounded by file endpoints.",
    )
    args = parser.parse_args()

    width_column = args.width_column or None
    min_chord_widths = None if args.min_chord_widths <= 0 else args.min_chord_widths
    bends, spectra = extract_bends_and_spectra(
        args.input,
        args.output,
        width=args.width,
        width_column=width_column,
        image_size=args.image_size,
        min_chord_widths=min_chord_widths,
        include_edge_bends=args.include_edge_bends,
    )
    print(f"Extracted {len(bends)} single bends")
    print(f"Saved spectra with shape {spectra.shape} to {args.output}")


if __name__ == "__main__":
    main()
