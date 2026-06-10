#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction and optional model classification.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--min-chord-widths", type=float, default=0.0)
    parser.add_argument("--endpoint-mode", choices=["ignore", "auto", "include"], default="auto")
    parser.add_argument("--include-edge-bends", action="store_true")
    args = parser.parse_args()

    extract_cmd = [
        sys.executable,
        "scripts/extract_single_bends.py",
        "--input",
        str(args.input),
        "--output",
        str(args.output / "bends"),
        "--min-chord-widths",
        str(args.min_chord_widths),
        "--endpoint-mode",
        args.endpoint_mode,
    ]
    if args.include_edge_bends:
        extract_cmd.append("--include-edge-bends")
    if args.width is not None:
        extract_cmd.extend(["--width", str(args.width), "--width-column", ""])
    subprocess.check_call(extract_cmd)

    if args.model is not None:
        classify_cmd = [
            sys.executable,
            "scripts/classify_single_bends.py",
            "--spectra",
            str(args.output / "bends" / "spectra.npy"),
            "--model",
            str(args.model),
            "--output",
            str(args.output / "classification"),
        ]
        subprocess.check_call(classify_cmd)


if __name__ == "__main__":
    main()
