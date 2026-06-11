# GUI

The Streamlit GUI supports interactive single-bend extraction from a centerline file.

Run it from the repository root:

```bash
streamlit run app/streamlit_app.py
```

The app can:

1. load a CSV, TXT, or DAT centerline file;
2. detect candidate single bends from curvature sign-change inflection points;
3. display the selected bend and its normalized geometry;
4. compute the legacy-compatible CWT energy spectrum for the selected isolated bend;
5. optionally load the Zenodo autoencoder and cluster extracted bends in latent space.

## Autoencoder option

Download the Zenodo model first:

```bash
python scripts/download_model.py --output models
```

Then enable **Use Zenodo autoencoder for latent-space clustering** in the GUI. The default path is:

```text
models/Autoencoder_Meander_Bend.h5
```

The GUI encodes the extracted bend spectra with the autoencoder encoder and applies K-means in the latent space. The resulting cluster labels are unsupervised labels; compare them with the paper figures before assigning semantic names such as symmetric, upstream-skewed, or downstream-skewed.
