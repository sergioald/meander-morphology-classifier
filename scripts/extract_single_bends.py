#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from meander_morphology.pipeline import extract_bends_and_spectra


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract single bends and isolated CWT spectra from a centerline file.")
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
        "--endpoint-mode",
        choices=["ignore", "auto", "include"],
        default="auto",
        help="How to treat file endpoints as possible bend boundaries.",
    )
    parser.add_argument(
        "--endpoint-curvature-tolerance",
        type=float,
        default=0.10,
        help="Auto endpoint threshold as fraction of max absolute curvature.",
    )
    parser.add_argument(
        "--include-edge-bends",
        action="store_true",
        help="Backward-compatible alias for --endpoint-mode include.",
    )
    parser.add_argument(
        "--no-cwt-pad",
        action="store_true",
        help="Disable mirror padding before CWT. Padding is cropped and does not include adjacent bends.",
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
        endpoint_mode=args.endpoint_mode,
        endpoint_curvature_tolerance=args.endpoint_curvature_tolerance,
        cwt_pad=not args.no_cwt_pad,
    )
    endpoint_count = sum(row.uses_endpoint_boundary for row in bends)
    print(f"Extracted {len(bends)} single-bend candidates")
    print(f"Endpoint-boundary candidates: {endpoint_count}")
    print(f"Saved spectra with shape {spectra.shape} to {args.output}")


if __name__ == "__main__":
    main()
