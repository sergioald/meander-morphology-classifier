# Compound bend workflow

This workflow adds the second-stage classifier foundation for simple and compound meander units.

The single-bend workflow cuts bends at consecutive curvature inflection points. The compound workflow instead works at reach scale:

1. Read a centreline table with `x`, `y`, and optionally `width`.
2. Resample and smooth the centreline.
3. Compute curvature along arclength.
4. Compute a Mexican-hat CWT energy scalogram for the full reach.
5. At each arclength position, find the dominant local CWT ridge.
6. Move from the ridge toward lower pseudo-frequencies and identify the first trough.
7. Integrate energy over this ridge-to-trough corridor.
8. Normalise the corridor-energy signal along the reach.
9. Pick valleys in that signal as boundaries between simple/compound meander units.
10. Save one 64 x 64 CWT spectrum image per extracted unit.

## Command-line example

```bash
python scripts/extract_compound_bends.py \
  --input examples/example_centerline.csv \
  --output outputs/example_compound_bends \
  --width-column width
```

Useful parameters:

```bash
--meander-window-widths 22.0
--min-unit-widths 8.0
--valley-prominence 0.05
--points-per-width 25
```

`--meander-window-widths` controls the moving window used to stabilise ridge detection. The default value, 22 channel widths, follows the paper logic of using twice the lower single-meander wavelength estimate of approximately 11 channel widths.

## Outputs

```text
compound_bend_summary.csv
compound_spectra.npy
compound_spectra/compound_unit_0000.png
diagnostics/compound_segmentation_signal.csv
```

The summary file includes the detected unit limits, length, sinuosity, maturity index, estimated number of lobes, and whether the unit is classed as compound.

## Notes

This patch adds segmentation and spectrum generation only. It does not yet add the Performer/CNN compound autoencoder or latent-space evolution plots. Those should be added after the segmentation outputs are stable on the example and simulation files.
