from __future__ import annotations

import argparse
from pathlib import Path
import traceback

import numpy as np

from meander_morphology.model import build_encoder_from_autoencoder, load_autoencoder, prepare_images_for_keras


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Zenodo autoencoder loading and encoder extraction.")
    parser.add_argument("--model", type=Path, default=Path("models/Autoencoder_Meander_Bend.h5"))
    parser.add_argument("--latent-dim", type=int, default=2)
    args = parser.parse_args()

    print(f"Model path: {args.model}")
    print(f"Exists: {args.model.exists()}")
    if not args.model.exists():
        raise SystemExit("Model file not found. Run: python scripts/download_model.py --output models")

    try:
        autoencoder = load_autoencoder(args.model)
        print("Loaded autoencoder:", type(autoencoder), getattr(autoencoder, "name", "<unnamed>"))
        encoder = build_encoder_from_autoencoder(autoencoder, latent_dim=args.latent_dim)
        print("Recovered encoder:", type(encoder), getattr(encoder, "name", "<unnamed>"))
        print("Encoder output shape:", getattr(encoder, "output_shape", None))
        dummy = prepare_images_for_keras(np.zeros((2, 64, 64), dtype="float32"))
        encoded = encoder.predict(dummy, verbose=0)
        print("Dummy encoded shape:", np.asarray(encoded).shape)
        print("OK")
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
