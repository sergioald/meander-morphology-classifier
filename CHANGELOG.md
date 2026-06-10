# Changelog

## 0.1.3 - Synthetic data and evaluation utilities

- Added Kinoshita synthetic single-bend generation utilities.
- Added scripts for synthetic bend generation, Kinoshita parameter histograms and autoencoder reconstruction evaluation.
- Added synthetic-data documentation and lightweight tests.
- Updated package metadata to use an SPDX license string.

## 0.1.2 - Endpoint and isolated-spectrum corrections

- Added endpoint handling modes for single-bend extraction: `ignore`, `auto`, and `include`.
- Kept interior inflection points unmerged so single-bend extraction does not create compound-looking units.
- Added isolated CWT spectrum generation with mirror padding cropped back to the selected bend.
- Updated the Streamlit GUI to highlight the selected bend on the centerline and clarify endpoint-boundary candidates.
- Added tests for endpoint modes and CWT cropping.

## 0.1.1 - Single-bend extraction correction

- Changed filtering so short candidates are removed after extraction rather than by thinning inflection points first.

## 0.1.0 - Initial cleaned single-bend release

- Added package structure, scripts, docs, examples, GUI skeleton and tests.
