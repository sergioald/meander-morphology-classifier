# Changelog

## 0.1.0 - Initial cleaned single-bend release

- Added reusable package structure for meander geometry, curvature, bend extraction, CWT spectra, model loading and clustering.
- Added CLI scripts for model download, single-bend extraction and autoencoder-based classification.
- Added Streamlit GUI skeleton for interactive river-centerline analysis.
- Added documentation for the single-bend workflow, model record and data format.
- Added small synthetic example centerline and unit tests.


## Unreleased

- Updated single-bend extraction so bend candidates are cut at consecutive true inflection points.
- Replaced inflection-point thinning with bend-level chord-length filtering to avoid merging neighbouring bends into compound-looking units.
- Added CLI and GUI controls for minimum chord length and edge partial bends.
