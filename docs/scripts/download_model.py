#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

from meander_morphology.model import ZENODO_MODEL_FILENAME, ZENODO_MODEL_MD5, ZENODO_MODEL_URL


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Zenodo autoencoder model.")
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / ZENODO_MODEL_FILENAME
    if target.exists() and not args.force:
        print(f"Model already exists: {target}")
        print(f"MD5: {md5sum(target)}")
        return

    print(f"Downloading {ZENODO_MODEL_URL}")
    with requests.get(ZENODO_MODEL_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    digest = md5sum(target)
    print(f"Saved: {target}")
    print(f"MD5: {digest}")
    if digest != ZENODO_MODEL_MD5:
        raise SystemExit(f"MD5 mismatch; expected {ZENODO_MODEL_MD5}")


if __name__ == "__main__":
    main()
