# Local graphical user interface

The repository includes a local Streamlit graphical user interface. It is launched from a terminal and opens in the user's browser, but it is intended as a local scientific GUI rather than a hosted webpage.

```bash
streamlit run app/streamlit_app.py
```

## Input centreline files

The GUI accepts CSV, TXT and DAT centreline files.

Preferred CSV format:

```text
x,y,width
0.0,0.0,100.0
1.0,0.2,100.0
```

The `width` column is optional. If a file contains a different width-column name, set the **Width column name** field in the sidebar before running the workflow. If no width values are detected, the GUI uses the fallback constant channel width.

Headerless TXT/DAT files are interpreted as numeric columns. The first two columns are used as `x` and `y`; if a fourth column is present, it is treated as width, matching the legacy research-file convention.

## Tabs

The GUI is organised into five tabs:

1. **Home**: describes the workflows and previews the uploaded centreline.
2. **Single-bend classifier**: extracts inflection-point single bends, displays their isolated CWT spectra and optionally applies the published single-bend autoencoder.
3. **Compound-bend workflow**: extracts CWT-energy meander units using reach-scale spectral segmentation and exports the compound unit table and spectra.
4. **Compound latent space**: encodes compound spectra using a local encoder or full autoencoder model from the separate model/data Zenodo record.
5. **Reproducibility**: summarises local model-file expectations and Zenodo separation between software and model/data artefacts.

## Compound CWT plots

The compound workflow shows two different CWT views:

- the **reach-scale CWT energy** used to identify compound-unit boundaries; and
- the selected unit's **64 x 64 model CWT image**, which is the same type of grayscale spectrum exported by the command-line workflow and passed to the autoencoder.

These two plots are expected to look different because they represent different stages of the workflow.

## Model files

The software Zenodo record should not include large trained model files. For compound latent-space inference, place the separately archived model/data files under `models/`, for example:

```text
models/compound_autoencoder.h5
models/world_latent_cloud.npy
```

The GUI defaults to `models/compound_autoencoder.h5` because it is currently the most robust option across the tested Windows/Anaconda setup. If a native encoder-only model is available and loads correctly in the local TensorFlow/Keras environment, tick **Model is encoder-only** and provide that encoder path.

## Recommended citation wording

In the manuscript, refer to the interface as a *local graphical user interface* or *local Streamlit-based graphical user interface*, not as a webpage. Suggested wording:

> The archived software release includes a local Streamlit-based graphical user interface that allows users to upload river centreline files, extract single and compound meander units, compute curvature-spectrum images, and encode compound units into the trained latent space when the Zenodo model files are provided.


## GUI polish notes

The compound latent-space tab is intentionally inference-only. The latent dimension is fixed at the trained 2-D embedding and the batch size is fixed internally; the GUI does not expose training-from-zero controls.

The compound workflow tab now shows a compact legacy-style reach diagnostic inspired by the research scripts: reach-scale CWT energy, the integrated corridor-energy signal, curvature, and the centreline with unit boundaries. The separate model CWT image remains the 64 x 64 grayscale image used as autoencoder input.
## GUI polish notes for paper/Zenodo release

The GUI separates display diagnostics from model inputs:

- The reach-scale CWT diagnostic is a legacy-style visual check for compound-unit segmentation. The frequency display range can be widened from the GUI without changing the segmentation calculation.
- The selected-unit model CWT image is the exact 64 x 64 array passed to the trained compound autoencoder. An enhanced preview is shown beside it only for human visibility.
- The compound latent-space plot uses a simple legend: reference meander cloud, detected meander units, and selected unit. It does not use `n_lobes` as the colour scale by default.
- The single-bend classifier shows a visible latent-space cluster plot and cluster-size table when the optional single-bend autoencoder is available locally.



## GUI display notes

The compound latent-space panel can optionally annotate detected units with their unit numbers. The reference cloud is shown only as context, while the selected detected unit is highlighted separately.

The compound CWT image shown in the workflow tab has two display modes. The enhanced preview is intended for visual inspection. The exact 64 x 64 image is the reproducible model input passed to the compound autoencoder; it uses pixel axes because it is a resized array rather than a physical CWT diagnostic figure.

The reach-scale CWT diagnostic is a display-only diagnostic. The maximum displayed frequency can be adjusted in the GUI without changing the underlying compound segmentation.


## Compound model CWT image convention

The compound-bend model-input image is different from the reach-scale CWT diagnostic. The reach-scale CWT is a physical/diagnostic plot used to inspect segmentation. The compound model image is a resized 64 x 64 training-style array generated from one extracted unit's curvature signal. It follows the legacy training-image polarity: white background with dark high-energy CWT structures. The GUI enhanced preview is for visual inspection only; the exact 64 x 64 array is the image passed to the compound autoencoder.

## Single-bend Zenodo model download

The single-bend classifier tab can download the public Zenodo autoencoder directly into `models/Autoencoder_Meander_Bend.h5`. This file remains ignored by Git and should not be committed. The GUI verifies the downloaded file with the MD5 checksum stored in `meander_morphology.model`.


## Compound model-image axes

The compound model-image preview uses the same 64 x 64 array passed to the autoencoder. The horizontal axis is displayed as `S_bend/S_bend,max` to match the training plots. The vertical axis is a CWT scale index because the resized model input does not preserve the original physical frequency or period coordinate.

