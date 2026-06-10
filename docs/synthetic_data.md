# Synthetic Kinoshita data

The original single-bend workflow used synthetic Kinoshita meanders to populate a broad set of bend geometries before encoding CWT energy spectra with the autoencoder. The cleaned repository now includes a reproducible, smaller-scale version of that generator for examples, tests and retraining experiments.

The generator follows the legacy research scripts:

- wavelength is derived from a gamma-distributed wavenumber sample;
- `theta_1` controls the first-order meander amplitude;
- `theta_3r` is the third-order fattening term;
- `theta_3i` is the third-order skewing term;
- curvature is computed analytically from the Kinoshita angle equation.

## Generate a small synthetic dataset

```bash
python scripts/generate_synthetic_bends.py \
  --output outputs/synthetic_bends \
  --n-bends 100 \
  --save-png
```

This creates:

```text
outputs/synthetic_bends/
  x_bends.npy
  y_bends.npy
  s_bends.npy
  curvature_bends.npy
  theta_bends.npy
  spectra.npy
  synthetic_bend_metadata.csv
  spectra_png/                # optional PNG spectra
```

The generated `spectra.npy` can be passed directly to the classification and autoencoder-evaluation scripts.

## Plot the parameter distributions

```bash
python scripts/plot_kinoshita_parameters.py \
  --output outputs/kinoshita_parameters
```

This saves a parameter table and histogram figure for the wavelength, `theta_1`, `theta_3r` and `theta_3i` distributions.

## Evaluate a downloaded autoencoder on spectra

```bash
python scripts/download_model.py --output models
python scripts/evaluate_autoencoder.py \
  --model models/Autoencoder_Meander_Bend.h5 \
  --spectra outputs/synthetic_bends/spectra.npy \
  --output outputs/autoencoder_evaluation
```

The evaluation script reports reconstruction MSE/MAE and writes per-sample reconstruction errors.

## Full research-scale generation

The legacy scripts used a Cartesian product of parameter arrays. With the original default of 77 values per Kinoshita parameter and a single median wavelength, the full grid contains 456,533 parameter combinations. The cleaned code exposes `legacy_parameter_grid()` for reproducibility, but the public scripts default to independent random samples so the examples remain lightweight.
