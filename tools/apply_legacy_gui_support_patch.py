from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
APP = ROOT / "app" / "streamlit_app.py"
COMPOUND_MODEL = ROOT / "src" / "meander_morphology" / "compound_model.py"


PYPROJECT_REPLACEMENT = '''[project.optional-dependencies]
deep-learning = ["tensorflow>=2.13"]
wavelets = ["PyWavelets>=1.5"]
gui = ["streamlit>=1.30"]
legacy-gui = [
  "streamlit==1.18.1",
  "altair==4.2.2",
  "pyarrow==10.0.1",
]
legacy-deep-learning = [
  "tensorflow==2.10.1",
  "keras==2.10.0",
  "numpy==1.23.5",
  "protobuf==3.19.6",
  "h5py==3.7.0",
]
dev = ["pytest>=7.4", "ruff>=0.5"]
'''


def patch_pyproject() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    if "legacy-deep-learning" in text and "legacy-gui" in text:
        print("pyproject.toml already has legacy extras")
        return

    start = text.find("[project.optional-dependencies]")
    end = text.find("\n[tool.setuptools.packages.find]", start)
    if start == -1 or end == -1:
        raise RuntimeError("Could not locate optional dependency block in pyproject.toml")

    text = text[:start] + PYPROJECT_REPLACEMENT + text[end:]
    PYPROJECT.write_text(text, encoding="utf-8")
    print("Patched pyproject.toml with legacy extras")


def patch_streamlit_app() -> None:
    text = APP.read_text(encoding="utf-8")

    # Streamlit 1.18 compatibility. Newer app versions used width="stretch"/"content".
    text = text.replace(', width="stretch"', ", use_container_width=True")
    text = text.replace(', width="content"', "")

    if "def _runtime_version_table()" not in text:
        marker = '''def _to_npy_download(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, np.asarray(array))
    buffer.seek(0)
    return buffer.getvalue()
'''
        addition = marker + r'''

def _runtime_version_table() -> pd.DataFrame:
    """Return local runtime versions for GUI/model compatibility checks."""
    rows = []
    rows.append({"package": "python", "version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"})
    rows.append({"package": "streamlit", "version": getattr(st, "__version__", "unknown")})
    rows.append({"package": "numpy", "version": getattr(np, "__version__", "unknown")})
    rows.append({"package": "pandas", "version": getattr(pd, "__version__", "unknown")})
    try:
        import tensorflow as tf
        rows.append({"package": "tensorflow", "version": getattr(tf, "__version__", "not installed")})
        rows.append({"package": "keras", "version": getattr(tf.keras, "__version__", "tf.keras")})
    except Exception as exc:
        rows.append({"package": "tensorflow", "version": f"not available: {exc}"})
    try:
        import h5py
        rows.append({"package": "h5py", "version": getattr(h5py, "__version__", "unknown")})
    except Exception as exc:
        rows.append({"package": "h5py", "version": f"not available: {exc}"})
    return pd.DataFrame(rows)


def _latent_scale_warning(latent_table: pd.DataFrame, background: np.ndarray | None) -> str | None:
    """Warn when encoded units are much smaller in scale than the background cloud."""
    if background is None:
        return None
    if not {"latent_1", "latent_2"}.issubset(latent_table.columns):
        return None
    detected = latent_table.loc[:, ["latent_1", "latent_2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    detected = detected[np.isfinite(detected).all(axis=1)]
    background = np.asarray(background, dtype=float)
    if detected.size == 0 or background.ndim != 2 or background.shape[1] < 2:
        return None
    bg = background[:, :2]
    bg = bg[np.isfinite(bg).all(axis=1)]
    if bg.size == 0:
        return None
    detected_std = float(np.nanstd(detected))
    background_std = float(np.nanstd(bg))
    if not np.isfinite(detected_std) or not np.isfinite(background_std) or background_std <= 0:
        return None
    ratio = detected_std / background_std
    if ratio < 0.01:
        return (
            "Encoded latent coordinates are less than 1% of the background-cloud scale. "
            "Do not trust the latent interpretation until the model file, encoder setting, "
            "background cloud and TensorFlow/Keras environment have been checked."
        )
    return None
'''
        if marker not in text:
            raise RuntimeError("Could not find _to_npy_download block in app/streamlit_app.py")
        text = text.replace(marker, addition)
        print("Added runtime/version diagnostics to app/streamlit_app.py")

    if "This tab loads archived TensorFlow/Keras model files" not in text:
        old = '''with latent_tab:
    st.subheader("Compound latent space")
    st.caption("Encode compound spectra with the separately archived model files.")
'''
        new = old + '''    st.warning(
        "This tab loads archived TensorFlow/Keras model files. For local model-backed "
        "inference, use the pinned environment in environment-legacy-gui.yml. If loading "
        "fails with TFOpLambda, or if latent coordinates collapse near zero while the "
        "background cloud spans much larger values, the environment/model combination "
        "is not validated."
    )
    with st.expander("Runtime compatibility check"):
        st.dataframe(_runtime_version_table(), use_container_width=True)
        st.caption(
            "Expected local legacy stack: Python 3.10, TensorFlow 2.10.1, Keras 2.10.0, "
            "NumPy 1.23.5, Streamlit 1.18.1."
        )
'''
        if old not in text:
            raise RuntimeError("Could not find latent_tab header block")
        text = text.replace(old, new)
        print("Added legacy environment warning to compound latent tab")

    if "_latent_scale_warning(latent_table, background)" not in text:
        old = '''            with lat_col:
                st.pyplot(plot_compound_latent(latent_table, background_latent=background, selected_id=selected, show_unit_labels=show_compound_latent_labels), clear_figure=True)
'''
        new = '''            scale_warning = _latent_scale_warning(latent_table, background)
            if scale_warning:
                st.error(scale_warning)
            with lat_col:
                st.pyplot(
                    plot_compound_latent(
                        latent_table,
                        background_latent=background,
                        selected_id=selected,
                        show_unit_labels=show_compound_latent_labels,
                    ),
                    clear_figure=True,
                )
'''
        if old in text:
            text = text.replace(old, new)
            print("Added latent scale warning to default latent plot block")
        else:
            marker = '''            with lat_col:
                st.pyplot(
                    plot_compound_latent(
'''
            insert = '''            scale_warning = _latent_scale_warning(latent_table, background)
            if scale_warning:
                st.error(scale_warning)
'''
            if marker in text:
                text = text.replace(marker, insert + marker)
                print("Added latent scale warning to patched latent plot block")
            else:
                print("Could not add latent scale warning automatically; plotting block may already be customised")

    APP.write_text(text, encoding="utf-8")
    print("Patched app/streamlit_app.py")


def patch_compound_model() -> None:
    text = COMPOUND_MODEL.read_text(encoding="utf-8")
    if "Use environment-legacy-gui.yml" in text:
        print("compound_model.py already has legacy-load guidance")
        return

    old = '''    if model_is_encoder:
        return tf.keras.models.load_model(model_path, compile=False)
'''
    new = '''    if model_is_encoder:
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the encoder-only model. If this is a legacy .h5 file "
                "and the error mentions TFOpLambda or unknown layers, use the pinned "
                "local environment in environment-legacy-gui.yml. Do not install "
                "the modern [gui,deep-learning] extras for legacy model inference."
            ) from exc
'''
    if old not in text:
        print("Could not find direct encoder load block in compound_model.py; leaving it unchanged")
        return

    text = text.replace(old, new)
    COMPOUND_MODEL.write_text(text, encoding="utf-8")
    print("Patched compound_model.py with clearer legacy-load error")


def main() -> None:
    if not PYPROJECT.exists() or not APP.exists() or not COMPOUND_MODEL.exists():
        raise SystemExit("Run this from an extracted patch inside the repository root.")

    patch_pyproject()
    patch_streamlit_app()
    patch_compound_model()

    print()
    print("Done. Now run:")
    print("  python -m compileall app/streamlit_app.py src scripts")
    print("  conda env create -f environment-legacy-gui.yml")
    print("  conda activate meander-gui-tf210")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
