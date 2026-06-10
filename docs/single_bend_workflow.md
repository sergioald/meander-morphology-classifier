# Single-bend workflow

The single-bend workflow follows the Water Resources Research paper:

1. Load a river centerline.
2. Smooth and resample the centerline.
3. Compute tangent angle and curvature.
4. Detect inflection points from curvature sign changes.
5. Extract single bends between consecutive inflection points.
6. Normalize each bend by channel width and rotate it into a common frame.
7. Convert bend curvature into a CWT energy-spectrum image.
8. Encode the image with the trained autoencoder.
9. Cluster latent coordinates into bend-shape groups.

The publication interprets the three main groups as symmetric, downstream-skewed and upstream-skewed bends. In this cleaned codebase, cluster numbering is kept explicit because K-means label ordering can change between datasets.
