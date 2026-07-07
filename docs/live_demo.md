# Live demo

A lightweight public Streamlit demo is available here:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://meander-morphology-classifier.streamlit.app/)

Direct link:

```text
https://meander-morphology-classifier.streamlit.app/
```

## Purpose

The live demo is designed for recruiters, collaborators and reviewers who want to try the repository without cloning it or creating a local conda environment.

It provides a one-click demonstration of the public, lightweight part of the workflow:

- synthetic public-safe centreline data by default;
- optional upload of small CSV/TXT/DAT centreline files;
- compound-bend extraction;
- CWT-energy segmentation diagnostics;
- downloadable compound-unit summary CSV files.

## What the hosted demo includes

The hosted demo supports:

- centreline loading from the built-in synthetic example or from a user-uploaded file;
- representative-width handling using either file width values or a fallback width;
- compound meander-unit extraction from curvature/CWT-energy structure;
- visualisation of detected units on the centreline;
- visualisation of the segmentation energy signal;
- tabular unit summaries;
- CSV export for lightweight reproducibility checks.

## What the hosted demo intentionally excludes

The hosted demo does **not** download or load the Zenodo autoencoder model files.

This is intentional. It keeps the public deployment lightweight, fast to start and suitable for Streamlit Community Cloud.

The hosted demo does **not** perform:

- TensorFlow/Keras model loading;
- single-bend autoencoder inference;
- compound latent-space encoding with archived model files;
- download of large Zenodo model/data artefacts;
- full reproduction of the associated manuscript figures;
- scientific re-validation against the original full river datasets.

## Full local workflow

The full model-backed workflow remains available in the local repository.

After installing the repository and downloading the required model artefacts, run:

```bash
streamlit run app/streamlit_app.py
```

The full local GUI supports the broader workflow documented in the main README, including model-backed single-bend and compound latent-space analysis.

## Deployment configuration

The hosted app is deployed from:

```text
Repository: sergioald/meander-morphology-classifier
Branch: main
Main file path: app.py
Live app: https://meander-morphology-classifier.streamlit.app/
```

The deployment uses:

```text
requirements.txt
.streamlit/config.toml
```

The public demo entry point is:

```text
app.py
```

The full local research GUI remains:

```text
app/streamlit_app.py
```

## Maintenance notes

When updating the demo, keep it lightweight:

- avoid adding TensorFlow/Keras to the hosted `requirements.txt`;
- avoid automatic Zenodo model downloads in `app.py`;
- keep the built-in example synthetic and public-safe;
- keep uploaded files optional and small;
- keep generated outputs downloadable but not committed to Git.

For model-backed workflows, use the local GUI and model files described in the README.
