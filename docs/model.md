# Autoencoder model

The single-bend autoencoder is hosted on Zenodo:

- Record: https://doi.org/10.5281/zenodo.13913710
- File: `Autoencoder_Meander_Bend.h5`
- Resource type: model

Download with:

```bash
python scripts/download_model.py --output models/
```

The model is used to encode 64 × 64 grayscale CWT-spectrum images into a two-dimensional latent representation. Clustering in this latent space is used to group single-bend morphologies.

The model record has its own license and citation requirements. Cite both the paper and the model record when using it.
