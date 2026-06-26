#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from meander_morphology.compound_pipeline import extract_compound_bends_and_spectra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract simple/compound meander units from a centreline using CWT-energy valley segmentation."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV/TXT/DAT centreline file.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--width", type=float, default=None, help="Constant channel width if no width column is present.")
    parser.add_argument("--width-column", default="width", help="Width column name, or empty string to ignore.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--points-per-width", type=int, default=25)
    parser.add_argument("--meander-window-widths", type=float, default=22.0)
    parser.add_argument("--min-unit-widths", type=float, default=8.0)
    parser.add_argument("--valley-prominence", type=float, default=0.05)
    args = parser.parse_args()

    width_column = args.width_column or None
    units, spectra = extract_compound_bends_and_spectra(
        args.input,
        args.output,
        width=args.width,
        width_column=width_column,
        image_size=args.image_size,
        points_per_width=args.points_per_width,
        meander_window_widths=args.meander_window_widths,
        min_unit_widths=args.min_unit_widths,
        valley_prominence=args.valley_prominence,
    )
    compound_count = sum(unit.is_compound for unit in units)
    print(f"Extracted {len(units)} CWT-energy meander units")
    print(f"Compound/complex units: {compound_count}")
    print(f"Saved compound spectra with shape {spectra.shape} to {args.output}")


if __name__ == "__main__":
    main()
