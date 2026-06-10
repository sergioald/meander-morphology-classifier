# GUI

The GUI is implemented in Streamlit to keep the project easy to run and easy to demonstrate.

```bash
python -m pip install -e ".[gui]"
streamlit run app/streamlit_app.py
```

The first GUI version supports:

- uploading a centerline file;
- previewing the planform;
- extracting single bends;
- viewing bend metadata;
- viewing normalized bend shapes;
- viewing CWT energy spectra.

A later version can add direct model loading, latent-space visualization and compound-bend workflows.
