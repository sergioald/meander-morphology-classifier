# Changelog

## 0.1.0

- Added compatibility loader for the Zenodo H5 autoencoder when TensorFlow/Keras rejects legacy `groups=1` convolution metadata.
- Added GUI option to use the Zenodo autoencoder for latent-space clustering.
- Removed the small diagnostic spectrum-thumbnail image from the GUI.
- Simplified the logarithmic-energy GUI label.
- Kept the full CWT contour plot for inspecting the selected isolated bend spectrum.
- Added synthetic Kinoshita bend generation and related documentation.
- Added legacy-compatible CWT calculation for isolated single bends.
- Added endpoint handling controls for single-bend extraction.
- Added command-line scripts, tests, documentation, and CI configuration.
