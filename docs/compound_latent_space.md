# Compound-bend latent-space encoding

This workflow is the second stage of the compound/multiple-loop classifier.
It takes the spectra produced by `scripts/extract_compound_bends.py` and maps
those spectra into the latent space of a trained compound-bend autoencoder or
encoder.

## Inputs

Required inputs:

```text
outputs/example_compound_bends/compound_spectra.npy
models/trained_autoencoder.h5       # full autoencoder, or
models/encoder_only.keras           # extracted encoder
```

Optional inputs:

```text
outputs/example_compound_bends/compound_bend_summary.csv
models/world_latent_cloud.npy
```

The background latent cloud is only used for plotting a reference cloud. It is
not needed for encoding.

## Install TensorFlow only when encoding is needed

The core segmentation workflow does not require TensorFlow. Encoding does:

```bash
python -m pip install -e ".[deep-learning]"
```

## Encode using a full autoencoder

```bash
python scripts/encode_compound_bends.py \
  --spectra outputs/example_compound_bends/compound_spectra.npy \
  --summary outputs/example_compound_bends/compound_bend_summary.csv \
  --model models/trained_autoencoder.h5 \
  --output outputs/example_compound_bends/encoded \
  --latent-layer-name Latent_Space \
  --plot
```

## Encode using an extracted encoder

```bash
python scripts/encode_compound_bends.py \
  --spectra outputs/example_compound_bends/compound_spectra.npy \
  --summary outputs/example_compound_bends/compound_bend_summary.csv \
  --model models/encoder_only.keras \
  --model-is-encoder \
  --output outputs/example_compound_bends/encoded \
  --plot
```

## Outputs

```text
compound_latent.npy
compound_latent.csv
compound_latent_space.png     # only with --plot
```

The CSV keeps the `unit_id` column. If a compound-bend summary CSV is provided,
all segmentation metadata are merged with the latent coordinates.

## Notes

Large model files, exported encoders, latent clouds, and generated outputs should
remain outside Git unless intentionally published as release assets or Zenodo
records.
