# Meander Morphology Classifier

[![Tests](https://github.com/sergioald/meander-morphology-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/sergioald/meander-morphology-classifier/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21134675.svg)](https://doi.org/10.5281/zenodo.21134675)
[![Article DOI](https://img.shields.io/badge/article-10.1029%2F2024WR037583-blue)](https://doi.org/10.1029/2024WR037583)
[![Compound model/data DOI](https://img.shields.io/badge/compound%20model%2Fdata-10.5281%2Fzenodo.20845480-blue)](https://doi.org/10.5281/zenodo.20845480)
[![Single-bend model DOI](https://img.shields.io/badge/single--bend%20model-10.5281%2Fzenodo.13913710-blue)](https://doi.org/10.5281/zenodo.13913710)

Local and scriptable workflows for **curvature-spectral meander morphology analysis**, including:
- **single-bend extraction and optional latent-space clustering**
- **compound-bend segmentation from reach-scale CWT energy**
- **compound-spectrum encoding into a 2-D latent space**
- **a local Streamlit GUI for reproducible exploration and export**

This repository accompanies the manuscript associated with **A data-driven approach to discern the curvature spectral complexity of compound meander bends** and provides reproducible software for both the legacy single-bend workflow and the compound-bend extension.

---

## Visual overview

### Local GUI overview

<p align="center">
  <img src="docs/assets/readme_gui_overview.png" alt="Meander Morphology Classifier GUI overview" width="1000">
</p>

### Compound latent-space workflow

<p align="center">
  <img src="docs/assets/readme_compound_latent_space.png" alt="Compound-bend latent space view in the GUI" width="1000">
</p>

The GUI is intended for **local use** and exposes the same repository workflows available from the Python API and command-line scripts.

---

## What the repository does

### 1) Single-bend workflow
- reads a centreline from CSV/TXT/DAT
- computes curvature-based bend extraction from inflection points
- supports optional clustering/encoding using the separately archived **single-bend autoencoder**
- exports summary tables and bend-level diagnostics

### 2) Compound-bend workflow
- computes a **reach-scale continuous wavelet transform (CWT) energy signal**
- groups neighbouring lobes into larger compound meander units
- saves unit summaries and compound spectra
- optionally encodes compound spectra with the separately archived **compound autoencoder**
- visualises detected units in a **2-D latent space** against the world/reference meander cloud

---

## Installation

Clone the repository and install it in a fresh environment.

```bash
git clone https://github.com/sergioald/meander-morphology-classifier.git
cd meander-morphology-classifier
pip install -e .
```

If you prefer Conda:

```bash
conda create -n meander-morphology python=3.11
conda activate meander-morphology
pip install -e .
```

---

## Quick start

### Launch the local GUI

```bash
streamlit run app/streamlit_app.py
```

### Run the test suite

```bash
pytest
```

### Compile-check scripts and app

```bash
python -m compileall src scripts app
```

---

## Example script workflows

### Extract compound meander units

```bash
python scripts/extract_compound_bends.py   --input examples/example_centerline.csv   --output outputs/example_compound_bends   --width-column width
```

### Encode extracted compound spectra

```bash
python scripts/encode_compound_bends.py   --spectra outputs/example_compound_bends/compound_spectra.npy   --summary outputs/example_compound_bends/compound_bend_summary.csv   --model models/compound_autoencoder.h5   --output outputs/example_compound_bends/encoded   --latent-layer-name Latent_Space   --background-latent models/world_latent_cloud.npy   --plot
```

### Run the combined compound workflow

```bash
python scripts/run_compound_workflow.py   --input examples/example_centerline.csv   --output outputs/example_compound_workflow   --width-column width   --compound-model models/compound_autoencoder.h5   --background-latent models/world_latent_cloud.npy
```

---

## Model files and Zenodo records

Large model files are **not stored directly in Git** and should remain local after download.

### Software archive
- **Repository software DOI:** [10.5281/zenodo.21134675](https://doi.org/10.5281/zenodo.21134675)

### Compound-bend model/data archive
- **Compound model/data DOI:** [10.5281/zenodo.20845480](https://doi.org/10.5281/zenodo.20845480)
- expected files include the full trained autoencoder, extracted encoder, world/reference latent cloud, metadata, and model card material

### Single-bend legacy model archive
- **Single-bend model DOI:** [10.5281/zenodo.13913710](https://doi.org/10.5281/zenodo.13913710)
- this model is optional and primarily used for the legacy single-bend latent-space clustering workflow

The GUI includes convenience buttons for local download/placement of the external model files.

---

## Repository layout

```text
app/                        Streamlit GUI
scripts/                    Command-line workflows
src/meander_morphology/     Core package code
examples/                   Example centreline input(s)
tests/                      Pytest suite
docs/                       Documentation and reproducibility notes
models/                     Local-only model placeholders / README
```

---

## Reproducibility and scope

This repository is designed to support:
- transparent example workflows
- reproducible local analysis of centreline files
- export of CSV/NPY outputs for verification and reuse
- separation of code from large model/data archives via Zenodo

Please note:
- model performance and interpretation depend on using **consistent preprocessing**
- the compound autoencoder expects the same style of compound-spectrum inputs used in training
- local model files downloaded from Zenodo should **not** be committed back into the repository

---

## Citation

If you use this repository, please cite the software record and the associated article.

### Software
Lopez Dubon, S., Sgarabotto, A., & Lanzoni, S. (2026). *Meander Morphology Classifier* (Version 0.2.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21134675

### Associated article
Lopez Dubon, S., Sgarabotto, A., & Lanzoni, S. *A data-driven approach to discern the curvature spectral complexity of compound meander bends*. Water Resources Research. https://doi.org/10.1029/2024WR037583

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

---

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE).
