# Data format

## Centerline CSV

Preferred format:

```csv
x,y,width
0.0,0.0,100.0
100.0,10.0,100.0
```

The width column can be local width or a repeated representative reach width. If no width column exists, provide a constant width with `--width`.

## Legacy DAT files

Legacy research files often use numeric columns where columns 0 and 1 are x and y and column 3 is width. The loader supports this format.

## Generated outputs

Extraction creates:

```text
bend_summary.csv
spectra.npy
spectra/bend_0000.png
```

Classification creates:

```text
latent_coordinates.npy
cluster_centers.npy
cluster_labels.npy
classification.csv
latent_space.png
```

Large data and generated outputs should not be committed to Git.
