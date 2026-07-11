from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def describe_array(name: str, arr: np.ndarray) -> dict[str, float | str | tuple[int, ...]]:
    values = np.asarray(arr, dtype=float)
    finite = np.isfinite(values)
    out: dict[str, float | str | tuple[int, ...]] = {"name": name, "shape": tuple(values.shape)}
    if not finite.any():
        out.update({"min": "nan", "max": "nan", "mean": "nan", "std": "nan"})
        return out
    vals = values[finite]
    out.update(
        {
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
        }
    )
    return out


def print_dict(title: str, data: dict[str, object]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in data.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose compound encoder/model compatibility.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--spectra", type=Path, required=True)
    parser.add_argument("--background-latent", type=Path, default=None)
    parser.add_argument("--model-is-encoder", action="store_true")
    parser.add_argument("--latent-layer-name", default="Latent_Space")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import tensorflow as tf
    from meander_morphology.compound_model import load_compound_encoder, encode_spectra_with_encoder

    print_dict(
        "Environment",
        {
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "tf.keras",
            "model": str(args.model),
            "spectra": str(args.spectra),
            "model_is_encoder": args.model_is_encoder,
            "latent_layer_name": args.latent_layer_name,
        },
    )

    spectra = np.load(args.spectra)
    print_dict("Spectra", describe_array("compound_spectra", spectra))

    encoder = load_compound_encoder(
        args.model,
        model_is_encoder=args.model_is_encoder,
        latent_layer_name=args.latent_layer_name,
        latent_dim=2,
    )
    print("\nLoaded model")
    print("------------")
    print(f"input_shape: {getattr(encoder, 'input_shape', None)}")
    print(f"output_shape: {getattr(encoder, 'output_shape', None)}")

    latent = encode_spectra_with_encoder(encoder, spectra, batch_size=args.batch_size)
    print_dict("Encoded latent", describe_array("latent", latent))

    if args.background_latent is not None and args.background_latent.exists():
        background = np.load(args.background_latent)
        print_dict("Background latent", describe_array("background", background))

        latent_std = float(np.nanstd(latent[:, :2]))
        background_std = float(np.nanstd(background[:, :2]))
        ratio = latent_std / background_std if background_std > 0 else np.nan
        print("\nScale check")
        print("-----------")
        print(f"latent_std/background_std: {ratio:.6g}")
        if np.isfinite(ratio) and ratio < 0.01:
            print("WARNING: encoded latent scale is less than 1% of the background cloud scale.")
            print("This usually means the model, model layer, spectra generation, or environment is inconsistent.")

    out_csv = Path("outputs") / "compound_latent_diagnostic.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit_id": np.arange(latent.shape[0]), "latent_1": latent[:, 0], "latent_2": latent[:, 1]}).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
