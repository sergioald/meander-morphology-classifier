# Legacy script mapping

The initial cleaned implementation was informed by the uploaded research scripts:

- `Extract_Data_art.py`, `Extract_Data_art_fig1.py`, `Extract_Data_Sergio_or.py`, `Extract_Data_Sergio_or_2.py`: centerline processing, curvature calculation, inflection-point detection, bend extraction and CWT-spectrum generation.
- `Multiple_Cluster_3.py`: autoencoder loading, encoder/decoder extraction, latent-space encoding and K-means clustering.
- `Plot_bar.py`: paper plotting utilities for cluster statistics and Ucayali time series.
- `Fig_1.py`: figure-specific visualization code.

The public package avoids hardcoded local paths and separates reusable functions from paper-figure scripts.
