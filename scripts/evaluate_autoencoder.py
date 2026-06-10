#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from meander_morphology.latent import reconstruct_spectra
from meander_morphology.model import load_autoencoder, prepare_images_for_keras


def _load_spectra(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Expected a .npy spectra file")
    return np.load(path)


def _metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    original = np.asarray(original, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    diff = original - reconstructed
    return {
        "n_samples": int(original.shape[0]),
        "mse_mean": float(np.mean(diff**2)),
        "mse_median": float(np.median(np.mean(diff**2, axis=(1, 2, 3)))),
        "mae_mean": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an autoencoder on CWT spectra.")
    parser.add_argument("--model", type=Path, required=True, help="Keras .h5 autoencoder model")
    parser.add_argument("--spectra", type=Path, required=True, help="spectra.npy file")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--save-reconstructions",
        action="store_true",
        help="Save reconstructed spectra as reconstructions.npy",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    spectra = prepare_images_for_keras(_load_spectra(args.spectra))
    model = load_autoencoder(args.model)
    reconstructed = reconstruct_spectra(model, spectra, batch_size=args.batch_size)

    metrics = _metrics(spectra, reconstructed)
    with open(args.output / "autoencoder_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(args.output / "autoencoder_metrics.csv", index=False)

    sample_mse = np.mean((spectra - reconstructed) ** 2, axis=(1, 2, 3))
    pd.DataFrame({"sample_id": np.arange(len(sample_mse)), "mse": sample_mse}).to_csv(
        args.output / "sample_reconstruction_error.csv",
        index=False,
    )

    if args.save_reconstructions:
        np.save(args.output / "reconstructions.npy", reconstructed)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
