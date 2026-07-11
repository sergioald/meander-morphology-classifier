# Local legacy model environment

The public repository supports two different usage modes.

## 1. Modern public mode

Use this mode for the lightweight Streamlit Cloud demo, segmentation, tests, documentation and general repository development.

This mode does **not** load the archived TensorFlow encoder/autoencoder.

Typical use:

```bash
conda activate meander-morphology
streamlit run app.py
```

or, for the full local GUI without legacy model inference:

```bash
streamlit run app/streamlit_app.py
```

## 2. Local legacy model mode

Use this mode when you want the full local GUI to load the archived compound encoder or autoencoder files.

The archived `.h5` models were created with an older TensorFlow/Keras stack. Newer Keras versions can fail with errors such as:

```text
Unknown layer: 'TFOpLambda'
```

or can produce latent coordinates that are numerically inconsistent with the archived background latent cloud.

Use the pinned conda environment:

```bash
conda env create -f environment-legacy-gui.yml
conda activate meander-gui-tf210
streamlit run app/streamlit_app.py
```

## GUI settings

For an encoder-only file:

```text
Compound model path: models/encoder_only.h5
Model is encoder-only: checked
Background latent cloud path: models/world_latent_cloud.npy
```

For a full autoencoder file:

```text
Compound model path: models/compound_autoencoder.h5
Model is encoder-only: unchecked
Latent layer name: Latent_Space
Background latent cloud path: models/world_latent_cloud.npy
```

## Important warning

Do **not** create the legacy GUI environment using:

```bash
python -m pip install -e ".[gui,deep-learning]"
```

That command is for the modern optional dependency set and can pull a newer TensorFlow/Keras stack.

The legacy environment is intentionally pinned and local-only. It should not be used for the public Streamlit Cloud deployment.
