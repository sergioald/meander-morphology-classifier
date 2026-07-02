# Meander Morphology Classifier

[![Tests](https://github.com/sergioald/meander-morphology-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/sergioald/meander-morphology-classifier/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21134675.svg)](https://doi.org/10.5281/zenodo.21134675)
[![Article DOI](https://img.shields.io/badge/Article-10.1029%2F2024WR037583-blue)](https://doi.org/10.1029/2024WR037583)

Python research-software toolkit and Streamlit GUI for curvature-based meander bend classification using continuous wavelet-transform (CWT) spectra, autoencoder latent spaces and clustering.

This repository contains two complementary workflows:

1. the **single-bend workflow** from Lopez Dubon, Sgarabotto, and Lanzoni (2025), which classifies inflection-bounded meander bends into symmetric, downstream-skewed and upstream-skewed classes;
2. the **compound/complex-bend workflow** associated with the companion manuscript, which segments compound meander units from curvature spectral energy and encodes their CWT spectra into a two-dimensional latent space.

The single-bend workflow is associated with the peer-reviewed article:

> Associated article DOI: https://doi.org/10.1029/2024WR037583

The public single-bend autoencoder model is hosted on Zenodo:

> Lopez Dubon, S., Sgarabotto, A., & Lanzoni, S. (2024). *Autoencoder for Meander Bends Classification*. Zenodo. https://doi.org/10.5281/zenodo.13913710

The compound-bend autoencoder, extracted encoder, and processed world/real-river latent-space reference cloud are archived separately and should be downloaded into the local `models/` directory:

> Compound model/data Zenodo DOI: https://doi.org/10.5281/zenodo.20845480
>
> Software/GUI Zenodo DOI: https://doi.org/10.5281/zenodo.21134675

## Repository status

This is a cleaned, GitHub-ready implementation derived from research scripts. It separates reusable methods from plotting and paper-production scripts.

Large research data, generated spectra, full river datasets and trained model files are not committed by default. See `docs/model_files.md` and `docs/reproducibility.md`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

For GUI use:

```bash
python -m pip install -e ".[gui,deep-learning]"
```

For model inference without the GUI:

```bash
python -m pip install -e ".[deep-learning]"
```

## Workflow A — quick single-bend demo

```bash
python scripts/extract_single_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/example_bends \
  --width-column width
```

## Workflow B — single-bend classification using the Zenodo autoencoder

```bash
python scripts/download_model.py --output models/
python scripts/classify_single_bends.py \
  --spectra outputs/example_bends/spectra.npy \
  --model models/Autoencoder_Meander_Bend.h5 \
  --output outputs/example_classification
```

## Workflow C — use the GUI

```bash
streamlit run app/streamlit_app.py
```

The GUI lets a user upload a river centerline file, preview the planform, compute curvature, detect inflection-point bends, generate CWT spectra and, when the model is available, encode/classify the bends.

## Workflow D — generate synthetic Kinoshita bends

```bash
python scripts/generate_synthetic_bends.py --output outputs/synthetic_bends --n-bends 100 --save-png
python scripts/plot_kinoshita_parameters.py --output outputs/kinoshita_parameters
```

The synthetic generator is useful for lightweight examples, model checks and future retraining experiments. See `docs/synthetic_data.md`.

## Workflow E — extract compound/complex meander units

```bash
python scripts/extract_compound_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/example_compound_bends \
  --width-column width
```

This workflow computes a full-reach curvature CWT, follows the dominant local spectral ridge, integrates ridge-to-trough corridor energy, and uses low-energy valleys to define compound meander-unit boundaries.

Expected outputs:

```text
outputs/example_compound_bends/compound_bend_summary.csv
outputs/example_compound_bends/compound_spectra.npy
outputs/example_compound_bends/compound_spectra/*.png
outputs/example_compound_bends/diagnostics/compound_segmentation_signal.csv
```

## Workflow F — encode compound units into latent space

After downloading the compound model files into `models/`, run either the full autoencoder workflow:

```bash
python scripts/encode_compound_bends.py \
  --spectra outputs/example_compound_bends/compound_spectra.npy \
  --summary outputs/example_compound_bends/compound_bend_summary.csv \
  --model models/compound_autoencoder.h5 \
  --latent-layer-name Latent_Space \
  --background-latent models/world_latent_cloud.npy \
  --output outputs/example_compound_bends/encoded \
  --plot
```

or the preferred extracted-encoder workflow:

```bash
python scripts/encode_compound_bends.py \
  --spectra outputs/example_compound_bends/compound_spectra.npy \
  --summary outputs/example_compound_bends/compound_bend_summary.csv \
  --model models/encoder_only.keras \
  --model-is-encoder \
  --background-latent models/world_latent_cloud.npy \
  --output outputs/example_compound_bends/encoded \
  --plot
```

## Workflow G — one-command compound workflow

```bash
python scripts/run_compound_workflow.py \
  --input examples/example_centerline.csv \
  --output outputs/example_compound_full \
  --width-column width \
  --model models/compound_autoencoder.h5 \
  --latent-layer-name Latent_Space \
  --background-latent models/world_latent_cloud.npy \
  --plot
```

This command runs compound extraction, spectrum generation, latent encoding, and diagnostic plotting in one step.

## Input data format

The simplest input is a CSV file with columns:

```text
x,y,width
```

- `x`, `y`: centerline coordinates in metres or any consistent length unit.
- `width`: local or representative channel width. If absent, provide `--width` in the CLI.

## What is included

```text
src/meander_morphology/   reusable package code
scripts/                  command-line workflows
app/                      Streamlit GUI
examples/                 small synthetic/example data
docs/                     documentation and reproducibility notes
tests/                    unit tests
models/                   local location for downloaded model files
```

## What is not included

Large research data, generated spectra, full river datasets, `.h5` model files, `.keras` model files and `.npy` latent clouds are not committed by default.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
python -m compileall src scripts app
```

## Citation

If you use this repository, please cite the associated article and the archived software release.

Associated article:

> https://doi.org/10.1029/2024WR037583

Archived software release:

> https://doi.org/10.5281/zenodo.21134675

Model/data records:

> Single-bend autoencoder: https://doi.org/10.5281/zenodo.13913710
>
> Compound-bend autoencoder and latent reference cloud: https://doi.org/10.5281/zenodo.20845480

A `CITATION.cff` file is included for GitHub and Zenodo metadata. The trained compound autoencoder and processed latent reference cloud should be cited separately via the model/data Zenodo record.

## License

Code in this repository is released under the MIT License. The model and third-party datasets retain their own licenses; see `docs/model_files.md` and `docs/data_format.md`.
