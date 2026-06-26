# Model files and external Zenodo record

The GitHub repository does not commit trained model files or large processed arrays. They are intentionally kept outside Git so that the software can remain lightweight and so that the exact trained artefacts used in the manuscript can be cited separately.

The compound-bend model/data Zenodo record should contain:

```text
trained_autoencoder.h5
encoder_only.h5
encoder_only.keras
model_architecture.json
encoder_architecture.json
model_summary.txt
encoder_summary.txt
world_latent_cloud.npy
world_latent_cloud_metadata.json
model_card.md
```

The recommended inference file is `encoder_only.keras`, because it is smaller and avoids full-autoencoder deserialisation issues across TensorFlow/Keras versions. The full `trained_autoencoder.h5` is useful for reconstruction-based validation and historical compatibility.

## Local file placement

Place downloaded model files in the local `models/` directory:

```text
models/compound_autoencoder.h5
models/encoder_only.keras
models/world_latent_cloud.npy
```

These files are ignored by `.gitignore` and must not be committed.

## Running inference with the full autoencoder

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

## Running inference with the extracted encoder

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

## Paper and Zenodo citation placeholders

Replace the placeholders below once the records are reserved/published:

```text
Software Zenodo DOI: [SOFTWARE_ZENODO_DOI]
Model/data Zenodo DOI: [MODEL_DATA_ZENODO_DOI]
Paper DOI or preprint DOI: [PAPER_DOI_OR_PREPRINT_DOI]
```
