# Validation notes

This document summarises what is currently validated in the `meander-morphology-classifier` repository, what is intentionally outside the validation scope, and what should be checked before using the workflows for new research outputs.

## Validation scope

This repository is public research software derived from research workflows. The validation focus is on:

- reproducible execution of the main command-line workflows;
- correct handling of expected input data formats;
- lightweight checks of bend extraction and synthetic-data workflows;
- smoke testing of package imports and executable scripts;
- separation between reusable software, local model files, generated outputs and large research datasets;
- clear documentation of required external model/data artefacts.

## What is tested

The test suite and development checks are intended to verify that:

- the Python package can be imported;
- core functions run on small example or synthetic inputs;
- command-line scripts can be executed in a lightweight development environment;
- the repository structure remains usable after editable installation;
- source files compile without syntax errors;
- GUI and deep-learning dependencies remain optional rather than mandatory for all users.

## What is not fully tested

The repository does not provide full scientific re-validation of every result reported in the associated publications. In particular, the following are outside the lightweight public test scope:

- retraining of the autoencoder models from the original full datasets;
- full reproduction of all manuscript figures;
- exhaustive validation across all original river datasets;
- benchmarking against every modelling variant used during manuscript development;
- validation of third-party datasets or external model files not committed to the repository.

## Model and data boundaries

Large research data, generated spectra, trained `.h5` / `.keras` model files and latent reference clouds are not committed to GitHub by default.

This is intentional. The repository separates:

- public reusable code;
- small example or synthetic data;
- documentation and reproducibility notes;
- external model/data artefacts archived through Zenodo or handled locally.

Users should consult the model/data records cited in the README before attempting full model-based inference.

## Expected behaviour

For a correctly formatted centreline input file, the workflows should:

- read centreline coordinates and optional width information;
- compute curvature or curvature-derived representations;
- extract single-bend or compound-bend units depending on the workflow;
- generate spectra or summary outputs;
- optionally encode spectra using externally supplied trained model files;
- write reproducible tabular and visual outputs to the requested output directory.

## Known limitations

- Results depend on centreline quality, smoothing choices and representative-width assumptions.
- Autoencoder-based workflows require the correct external model files.
- The compound-bend workflow should be treated as a research workflow rather than a universally validated operational classifier.
- The repository is not a full hydrodynamic or morphodynamic solver.
- Classification outputs should be interpreted together with visual diagnostics and domain knowledge.

## Recommended checks before new use

Before using the repository for a new dataset, users should:

- verify coordinate units and centreline ordering;
- inspect the planform visually;
- check curvature and inflection-point detection;
- confirm width assumptions;
- run the workflow on a small subset first;
- review diagnostic plots before interpreting classifications;
- document any parameter changes used for publication-quality analysis.

## Development checks

Recommended local checks are:

```bash
python -m pip install -e ".[dev]"
pytest
python -m compileall src scripts app
```

For GUI or model-inference use, install the relevant optional dependencies described in the README.
