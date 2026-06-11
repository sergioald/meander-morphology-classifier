# Single-bend workflow

The single-bend workflow is:

1. Read a centerline table with `x`, `y` and optionally `width`.
2. Resample and smooth the centerline.
3. Compute intrinsic angle and curvature.
4. Detect interior inflection points from curvature sign changes.
5. Extract candidate single bends between consecutive inflection points.
6. Normalize each retained bend by width and rotate it to a common orientation.
7. Compute the CWT energy spectrum from the isolated normalized bend.
8. Encode the spectrum with the pre-trained autoencoder and cluster/interpret the latent coordinates.

## Important CWT detail

The CWT image should not be computed from a reach that includes neighbouring bends. The legacy research scripts use this order:

```text
selected single bend geometry
→ reflect the selected bend at both ends
→ recompute angle and curvature on the reflected three-bend geometry
→ crop back to the central selected bend
→ compute CWT on the cropped central-bend curvature
```

The reflected geometry is only a derivative-stabilisation step. It is removed before the CWT. This avoids intentionally adding adjacent bends to the spectrum while reducing curvature artefacts at the selected bend boundaries.

The GUI and extraction scripts now follow this order. Bright structures close to the left/right edge of a spectrum can still occur because a finite-length wavelet transform has edge/cone-of-influence effects, but they are not produced by including neighbouring centerline bends.
