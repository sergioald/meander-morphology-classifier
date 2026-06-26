# Compound-bend end-to-end workflow

This page documents the combined command-line workflow for the second-stage
compound/complex meander classifier.

The script performs the following steps:

```text
centreline CSV
-> reach-scale curvature
-> CWT-energy valley segmentation
-> compound meander-unit spectra
-> optional autoencoder latent encoding
-> optional latent diagnostics and plots
```

## Extraction only

```bash
python scripts/run_compound_workflow.py \
  --input examples/example_centerline.csv \
  --output outputs/example_compound_full \
  --width-column width
```

This writes:

```text
outputs/example_compound_full/compound_bend_summary.csv
outputs/example_compound_full/compound_spectra.npy
outputs/example_compound_full/compound_spectra/*.png
outputs/example_compound_full/diagnostics/compound_segmentation_signal.csv
```

## Extraction plus latent encoding

The model files are intentionally local-only and should not be committed.
Place them in `models/` or another ignored local folder, for example:

```text
models/compound_autoencoder.h5
models/world_latent_cloud.npy
```

Then run:

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

This additionally writes:

```text
outputs/example_compound_full/encoded/compound_latent.npy
outputs/example_compound_full/encoded/compound_latent.csv
outputs/example_compound_full/encoded/compound_latent_diagnostics.csv
outputs/example_compound_full/encoded/compound_latent_summary.json
outputs/example_compound_full/encoded/compound_latent_diagnostics.png
```

## Diagnostic columns

The diagnostic CSV keeps the original compound-unit metadata and appends:

- `latent_radius`: distance from the latent origin;
- `latent_angle_deg`: polar angle in the 2-D latent space;
- `complexity_class`: simple/compound class derived from `n_lobes`;
- `background_centroid_distance`: distance from the background latent-cloud centroid;
- `nearest_background_distance`: nearest-neighbour distance to the background cloud;
- `background_radius_percentile`: percentile of the unit radius relative to the background cloud.

These quantities are diagnostics only. They help inspect whether the encoded units
are concentrated in one part of the latent space or dispersed across the reference
world cloud, but they should not be interpreted directly as physical migration
rates.
