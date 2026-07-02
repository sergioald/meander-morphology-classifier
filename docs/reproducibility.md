# Reproducibility guide

This guide records the minimum commands needed to reproduce the example single-bend and compound-bend workflows from a fresh clone.

## 1. Environment

```bash
conda create -n meander-morphology python=3.11 -y
conda activate meander-morphology
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For model inference:

```bash
python -m pip install -e ".[deep-learning]"
```

## 2. Test the package

```bash
pytest
python -m compileall src scripts app
```

## 3. Single-bend workflow

```bash
python scripts/extract_single_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/example_bends \
  --width-column width
```

## 4. Compound-bend segmentation

```bash
python scripts/extract_compound_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/example_compound_bends \
  --width-column width
```

Expected key outputs:

```text
outputs/example_compound_bends/compound_bend_summary.csv
outputs/example_compound_bends/compound_spectra.npy
outputs/example_compound_bends/compound_spectra/*.png
outputs/example_compound_bends/diagnostics/compound_segmentation_signal.csv
```

## 5. Compound latent encoding

Download the compound model/data artefacts from the model Zenodo record (https://doi.org/10.5281/zenodo.20845480) and place them in `models/`.

Using the full autoencoder:

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

Using the extracted encoder:

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

## 6. One-command compound workflow

If the end-to-end runner is available, the full compound workflow can be run with:

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

## Notes

- Generated `outputs/` files are ignored by Git.
- Model files in `models/*.h5`, `models/*.keras`, and related binary artefacts are ignored by Git.
- The exact trained model/data artefacts used in the manuscript should be cited via the separate model/data Zenodo DOI: https://doi.org/10.5281/zenodo.20845480.
- Associated article DOI: https://doi.org/10.1029/2024WR037583.
