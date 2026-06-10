# Meander Morphology Classifier

Python research-software toolkit and Streamlit GUI for curvature-based meander bend classification using continuous wavelet-transform (CWT) spectra, autoencoder latent spaces and clustering.

This repository starts with the **single-bend workflow** from:

> Lopez Dubon, S., Sgarabotto, A., & Lanzoni, S. (2025). *A curvature-based framework for automated classification of meander bends*. Water Resources Research, 61, e2024WR037583. https://doi.org/10.1029/2024WR037583

The published workflow uses river centerlines, curvature, CWT energy spectra, a trained autoencoder and clustering to identify three single-bend classes:

- **C1**: symmetric bends
- **C2**: downstream-skewed bends
- **C3**: upstream-skewed bends

The public autoencoder model is hosted on Zenodo:

> Lopez Dubon, S., Sgarabotto, A., & Lanzoni, S. (2024). *Autoencoder for Meander Bends Classification*. Zenodo. https://doi.org/10.5281/zenodo.13913710

## Repository status

This is a cleaned, GitHub-ready implementation derived from research scripts. It separates reusable methods from plotting and paper-production scripts.

The first release focuses on the single-bend workflow. A later extension can add the compound/multiple-loop bend workflow from the follow-up paper.

## Workflows

### Workflow A — quick demo with synthetic/example data

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
python scripts/extract_single_bends.py --input examples/example_centerline.csv --output outputs/example_bends --width-column width
```

### Workflow B — use the Zenodo autoencoder model

```bash
python -m pip install -e ".[dev,deep-learning]"
python scripts/download_model.py --output models/
python scripts/classify_single_bends.py \
  --spectra outputs/example_bends/spectra.npy \
  --model models/Autoencoder_Meander_Bend.h5 \
  --output outputs/example_classification
```

### Workflow C — use the GUI

```bash
python -m pip install -e ".[gui,deep-learning]"
streamlit run app/streamlit_app.py
```

The GUI lets a user upload a river centerline file, preview the planform, compute curvature, detect inflection-point bends, generate CWT spectra and, when the model is available, encode/classify the bends.

## Input data format

The simplest input is a CSV file with columns:

```text
x,y,width
```

- `x`, `y`: centerline coordinates in metres or any consistent length unit.
- `width`: local or representative channel width. If absent, provide `--width` in the CLI.

Example:

```bash
python scripts/extract_single_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/example_bends \
  --width-column width
```

## What is included

```text
src/meander_morphology/   reusable package code
scripts/                  command-line workflows
app/                      Streamlit GUI
examples/                 small synthetic/example data
docs/                     documentation
tests/                    unit tests
models/                   location for downloaded model files
```

## What is not included

Large research data, generated spectra, full river datasets and `.h5` model files are not committed by default. Use `scripts/download_model.py` to obtain the public Zenodo model.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
python -m compileall src scripts app
```

## License

Code in this repository is released under the MIT License. The model and third-party datasets retain their own licenses; see `docs/model.md` and `docs/data_format.md`.
