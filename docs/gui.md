# Local graphical user interface

The repository includes a local Streamlit graphical user interface. It is launched from a terminal and opens in the user's browser, but it is intended as a local scientific GUI rather than a hosted webpage.

```bash
streamlit run app/streamlit_app.py
```

## Tabs

The GUI is organised into five tabs:

1. **Home**: describes the workflows and previews the uploaded centreline.
2. **Single-bend classifier**: extracts inflection-point single bends, displays their isolated CWT spectra and optionally applies the published single-bend autoencoder.
3. **Compound-bend workflow**: extracts CWT-energy meander units using reach-scale spectral segmentation and exports the compound unit table and spectra.
4. **Compound latent space**: encodes compound spectra using a local encoder or full autoencoder model from the separate model/data Zenodo record.
5. **Reproducibility**: summarises local model-file expectations and Zenodo separation between software and model/data artefacts.

## Model files

The software Zenodo record should not include large trained model files. For compound latent-space inference, place the separately archived model/data files under `models/`, for example:

```text
models/compound_autoencoder.h5
models/compound_autoencoder.h5
models/world_latent_cloud.npy
```

The GUI defaults to `models/compound_autoencoder.h5` because the encoder-only model is lighter and avoids unnecessary decoder loading during inference.

## Recommended citation wording

In the manuscript, refer to the interface as a *local graphical user interface* or *local Streamlit-based graphical user interface*, not as a webpage. Suggested wording:

> The archived software release includes a local Streamlit-based graphical user interface that allows users to upload river centreline files, extract single and compound meander units, compute curvature-spectrum images, and encode compound units into the trained latent space when the Zenodo model files are provided.
