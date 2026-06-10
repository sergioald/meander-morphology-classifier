# Single-bend workflow

The single-bend workflow follows the Water Resources Research paper:

1. read a river centerline with `x`, `y`, and optionally `width`;
2. smooth and resample the centerline;
3. compute tangent angle and curvature;
4. detect curvature sign-change inflection points;
5. extract bend candidates between consecutive inflection-point boundaries;
6. normalize each bend by width and rotate it to a common orientation;
7. compute an isolated CWT energy spectrum for each bend;
8. optionally encode the spectrum with the Zenodo autoencoder and cluster latent coordinates.

## Endpoint handling

Interior inflection points are never thinned before extraction. Removing interior inflections can merge neighbouring single bends into compound-looking units.

File endpoints are handled separately because a centerline file may start or end exactly at a true inflection point, or it may cut through a partial bend. The extractor provides three modes:

- `--endpoint-mode ignore`: use only interior sign-change inflections. This is strict and excludes endpoint candidates.
- `--endpoint-mode auto`: include an endpoint only if its absolute curvature is small relative to the reach. This is the GUI default.
- `--endpoint-mode include`: always use endpoints. This can include partial edge bends.

Example:

```bash
python scripts/extract_single_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/bends \
  --min-chord-widths 0 \
  --endpoint-mode auto
```

For publication-style filtering of short candidates, use a chord-length threshold such as:

```bash
python scripts/extract_single_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/bends_filtered \
  --min-chord-widths 5 \
  --endpoint-mode auto
```

## CWT spectrum isolation

The CWT spectrum is computed from the selected bend only. The default implementation mirror-pads the bend curvature before the CWT to reduce edge artefacts, then crops the energy matrix back to the original bend extent. This padding does not add adjacent bends to the saved spectrum.

To disable mirror padding:

```bash
python scripts/extract_single_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/bends_no_padding \
  --no-cwt-pad
```
