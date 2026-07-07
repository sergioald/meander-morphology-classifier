# Live demo deployment

This repository includes a lightweight public Streamlit demo in `app.py`.

The demo is designed for recruiters, collaborators and reviewers who want to try the project without cloning the repository or creating a conda environment.

## What the hosted demo includes

The hosted demo supports:

- synthetic public-safe centreline data by default;
- optional upload of small CSV/TXT/DAT centreline files;
- compound-bend extraction;
- CWT-energy segmentation diagnostics;
- downloadable compound-unit summary CSV files.

## What the hosted demo intentionally excludes

The hosted demo does **not** download or load the Zenodo autoencoder model files. This keeps deployment lightweight and avoids long cold-start times.

Model-backed workflows remain available in the full local repository through:

```bash
streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud settings

Use these deployment settings:

```text
Repository: sergioald/meander-morphology-classifier
Branch: main
Main file path: app.py
```

The app uses:

```text
requirements.txt
.streamlit/config.toml
```

## Suggested README badge after deployment

After Streamlit creates the app URL, add a badge near the top of `README.md`.

Replace `YOUR_APP_URL` with the final Streamlit URL:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL)
```

## Suggested README section

```markdown
## Try the live demo

A lightweight public demo is available online:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL)

The hosted demo uses synthetic/example centreline data and demonstrates compound-bend extraction and CWT-energy segmentation diagnostics without requiring a local install. The full autoencoder workflows require local Zenodo model files and are documented below.
```
