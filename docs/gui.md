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
