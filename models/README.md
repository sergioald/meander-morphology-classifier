# Models

Model files are not committed to this repository by default.

## Single-bend model

The public single-bend autoencoder from the first paper can be downloaded with:

```bash
python scripts/download_model.py --output models/
```

Expected file:

```text
models/Autoencoder_Meander_Bend.h5
```

Source:

> Lopez Dubon, S., Sgarabotto, A., & Lanzoni, S. (2024). Autoencoder for Meander Bends Classification. Zenodo. https://doi.org/10.5281/zenodo.13913710

## Compound-bend model files

The compound-bend workflow uses a separate model/data Zenodo record associated with the companion manuscript. Once published, download the compound autoencoder, extracted encoder, and world latent reference cloud from:

```text
[MODEL_DATA_ZENODO_DOI]
```

Recommended local filenames:

```text
models/compound_autoencoder.h5
models/encoder_only.keras
models/world_latent_cloud.npy
```

These files are ignored by Git and must not be committed.
