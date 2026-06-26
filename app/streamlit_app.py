from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from meander_morphology.bends import extract_single_bends
from meander_morphology.clustering import cluster_latent_space
from meander_morphology.compound import extract_compound_bends
from meander_morphology.compound_diagnostics import build_latent_diagnostics
from meander_morphology.compound_model import (
    encode_spectra_with_encoder,
    load_compound_encoder,
)
from meander_morphology.cwt import cwt_energy_from_geometry, energy_to_image, spectrum_image_from_geometry
from meander_morphology.io import read_centerline_table
from meander_morphology.latent import encode_spectra
from meander_morphology.model import build_encoder_from_autoencoder, load_autoencoder


@st.cache_resource(show_spinner=False)
def load_autoencoder_and_encoder(model_path: str, latent_dim: int = 2):
    autoencoder = load_autoencoder(model_path)
    encoder = build_encoder_from_autoencoder(autoencoder, latent_dim=latent_dim)
    return autoencoder, encoder


@st.cache_resource(show_spinner=False)
def load_cached_compound_encoder(
    model_path: str,
    *,
    model_is_encoder: bool,
    latent_layer_name: str,
    latent_dim: int,
):
    return load_compound_encoder(
        model_path,
        model_is_encoder=model_is_encoder,
        latent_layer_name=latent_layer_name,
        latent_dim=latent_dim,
    )


def _to_csv_download(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False).encode("utf-8")


def _to_npy_download(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, np.asarray(array))
    buffer.seek(0)
    return buffer.getvalue()


def read_uploaded_centerline(uploaded_file) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    return read_centerline_table(tmp_path)


def plot_centerline_with_bend(x: np.ndarray, y: np.ndarray, bend=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, linewidth=1.2, label="centerline")
    if bend is not None:
        ax.plot(bend.raw_x, bend.raw_y, linewidth=3, label=f"bend {bend.bend_id}")
        ax.scatter([bend.raw_x[0], bend.raw_x[-1]], [bend.raw_y[0], bend.raw_y[-1]], s=35, label="boundaries")
        local_apex = int(np.argmax(np.abs(bend.curvature)))
        if 0 <= local_apex < len(bend.raw_x):
            ax.scatter([bend.raw_x[local_apex]], [bend.raw_y[local_apex]], marker="x", s=70, label="apex")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_normalized_bend(bend):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(bend.x, bend.y, linewidth=2)
    ax.scatter([bend.x[0], bend.x[-1]], [bend.y[0], bend.y[-1]], s=35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / width")
    ax.set_ylabel("y / width")
    fig.tight_layout()
    return fig


def plot_curvature_and_spectrum(result):
    fig, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.contourf(
        result.s,
        result.periods,
        result.energy,
        levels=32,
        cmap="gray",
    )
    ax.invert_yaxis()
    ax.set_xlabel("s / smax")
    ax.set_ylabel("period")
    fig.colorbar(mesh, ax=ax, label="energy")
    fig.tight_layout()
    return fig


def plot_single_latent_space(latent: np.ndarray, labels: np.ndarray, selected_id: int):
    fig, ax = plt.subplots(figsize=(5, 4))
    if latent.shape[1] >= 2:
        scatter = ax.scatter(latent[:, 0], latent[:, 1], c=labels, s=45)
        ax.scatter(latent[selected_id, 0], latent[selected_id, 1], marker="x", s=120, linewidths=3)
        ax.set_xlabel("latent 1")
        ax.set_ylabel("latent 2")
        fig.colorbar(scatter, ax=ax, label="cluster")
    else:
        ax.scatter(np.arange(latent.shape[0]), latent[:, 0], c=labels, s=45)
        ax.scatter([selected_id], [latent[selected_id, 0]], marker="x", s=120, linewidths=3)
        ax.set_xlabel("bend ID")
        ax.set_ylabel("latent value")
    fig.tight_layout()
    return fig


def plot_compound_units_on_centerline(x: np.ndarray, y: np.ndarray, units, selected_id: int | None = None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, linewidth=1.0, label="centerline")
    for unit in units:
        linewidth = 3.0 if unit.unit_id == selected_id else 1.8
        alpha = 1.0 if unit.unit_id == selected_id else 0.55
        ax.plot(unit.raw_x, unit.raw_y, linewidth=linewidth, alpha=alpha)
        ax.scatter([unit.raw_x[0], unit.raw_x[-1]], [unit.raw_y[0], unit.raw_y[-1]], s=20, alpha=alpha)
        if len(unit.raw_x):
            mid = len(unit.raw_x) // 2
            ax.annotate(str(unit.unit_id), (unit.raw_x[mid], unit.raw_y[mid]), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("CWT-energy meander units")
    fig.tight_layout()
    return fig


def plot_compound_segmentation_signal(segmentation, units=None, selected_id: int | None = None):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(segmentation.s, segmentation.normalised_energy, linewidth=1.5, label="normalised corridor energy")
    for boundary in segmentation.boundary_indices:
        ax.axvline(segmentation.s[int(boundary)], linewidth=0.8, alpha=0.45)
    if units and selected_id is not None:
        unit = units[selected_id]
        ax.axvspan(unit.raw_s[0] + segmentation.s[unit.start_index], segmentation.s[unit.end_index], alpha=0.12)
    ax.set_xlabel("along-centreline distance")
    ax.set_ylabel("normalised energy")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_compound_latent(table: pd.DataFrame, background_latent: np.ndarray | None = None, selected_id: int | None = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    if background_latent is not None and background_latent.ndim == 2 and background_latent.shape[1] >= 2:
        ax.scatter(background_latent[:, 0], background_latent[:, 1], s=2, alpha=0.12, label="background latent cloud")
    if {"latent_1", "latent_2"}.issubset(table.columns):
        if "n_lobes" in table.columns:
            scatter = ax.scatter(table["latent_1"], table["latent_2"], c=table["n_lobes"], s=55, label="compound units")
            fig.colorbar(scatter, ax=ax, label="n_lobes")
        else:
            ax.scatter(table["latent_1"], table["latent_2"], s=55, label="compound units")
        for _, row in table.iterrows():
            ax.annotate(str(int(row.get("unit_id", 0))), (row["latent_1"], row["latent_2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        if selected_id is not None and selected_id < len(table):
            row = table.iloc[selected_id]
            ax.scatter([row["latent_1"]], [row["latent_2"]], marker="x", s=140, linewidths=3)
    ax.set_xlabel("latent_1")
    ax.set_ylabel("latent_2")
    ax.set_title("Compound-bend latent space")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def build_spectrum_images_for_single_model(bends) -> np.ndarray:
    images = []
    for bend in bends:
        result = cwt_energy_from_geometry(
            bend.x,
            bend.y,
            target_points=201,
            log_energy=False,
        )
        images.append(energy_to_image(result.energy, image_size=64))
    return np.asarray(images, dtype="float32")


def build_spectrum_images_for_compound_model(units, *, image_size: int = 64) -> np.ndarray:
    images = []
    for unit in units:
        images.append(spectrum_image_from_geometry(unit.x, unit.y, image_size=image_size, target_points=201))
    return np.asarray(images, dtype="float32")


st.set_page_config(page_title="Meander Morphology Classifier", layout="wide")
st.title("Meander Morphology Classifier")
st.caption(
    "Local Streamlit GUI for curvature-based single-bend classification and "
    "CWT-energy compound-bend latent-space analysis."
)

with st.sidebar:
    st.header("Input centreline")
    uploaded = st.file_uploader("Upload centreline CSV/TXT/DAT", type=["csv", "txt", "dat"])
    fallback_width = st.number_input("Fallback channel width", min_value=0.0, value=100.0, step=10.0)
    st.caption("Preferred CSV columns: x, y and optionally width.")

x = y = width_values = None
width_source = fallback_width
if uploaded is not None:
    try:
        x, y, width_values = read_uploaded_centerline(uploaded)
        width_source = width_values if width_values is not None else fallback_width
    except Exception as exc:
        st.error(f"Could not read centreline file: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

about_tab, single_tab, compound_tab, latent_tab, reproducibility_tab = st.tabs(
    [
        "Home",
        "Single-bend classifier",
        "Compound-bend workflow",
        "Compound latent space",
        "Reproducibility",
    ]
)

with about_tab:
    st.subheader("Workflow overview")
    st.write(
        "This local GUI exposes the repository workflows used for curvature-spectral "
        "meander morphology analysis. The single-bend tab follows the inflection-point "
        "workflow. The compound tabs use reach-scale CWT energy to group neighbouring "
        "lobes into larger meander units before optional autoencoder encoding."
    )
    st.markdown(
        """
        **Recommended use before publication/Zenodo:**

        1. Validate extraction on a small centreline file.
        2. Check the compound segmentation signal and unit table.
        3. Encode compound spectra only when the model files from the separate Zenodo record are available locally.
        4. Export CSV/NPY outputs for reproducibility checks.
        """
    )
    if x is not None:
        st.success(f"Loaded centreline with {len(x)} points.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, linewidth=1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Upload a centreline file from the sidebar to begin.")

with single_tab:
    st.subheader("Single-bend classifier")
    st.caption("Inflection-point extraction and optional legacy autoencoder clustering.")
    if x is None:
        st.info("Upload a centreline file from the sidebar to use this tab.")
    else:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            min_chord_widths = st.number_input(
                "Minimum bend chord length / width",
                min_value=0.0,
                value=0.0,
                step=0.5,
                help="Use 5--8 for publication-style filtering. Use 0 to keep all candidates.",
                key="single_min_chord_widths",
            )
        with col_b:
            endpoint_mode = st.selectbox(
                "Endpoint handling",
                options=["auto", "ignore", "include"],
                index=0,
                key="single_endpoint_mode",
            )
        with col_c:
            endpoint_tolerance = st.slider(
                "Auto endpoint curvature tolerance",
                min_value=0.0,
                max_value=0.30,
                value=0.10,
                step=0.01,
                key="single_endpoint_tolerance",
            )

        use_autoencoder = st.checkbox("Use single-bend Zenodo autoencoder", value=False)
        model_path = st.text_input(
            "Single-bend autoencoder path",
            value="models/Autoencoder_Meander_Bend.h5",
            disabled=not use_autoencoder,
        )
        latent_dim = st.number_input("Latent dimension", min_value=1, max_value=16, value=2, step=1, disabled=not use_autoencoder)
        n_clusters = st.number_input("Number of latent-space clusters", min_value=2, max_value=10, value=3, step=1, disabled=not use_autoencoder)

        min_filter = None if min_chord_widths <= 0 else float(min_chord_widths)
        bends = extract_single_bends(
            x,
            y,
            width=width_source,
            min_chord_widths=min_filter,
            endpoint_mode=endpoint_mode,
            endpoint_curvature_tolerance=endpoint_tolerance,
        )
        metadata = pd.DataFrame([bend.metadata() for bend in bends])
        latent = None
        cluster_labels = None

        if use_autoencoder and bends:
            model_file = Path(model_path)
            if not model_file.exists():
                st.warning(f"Model file not found: {model_file}")
            elif len(bends) < int(n_clusters):
                st.warning("The number of bends must be at least the requested number of clusters.")
            else:
                try:
                    with st.spinner("Encoding single bends..."):
                        _, encoder = load_autoencoder_and_encoder(str(model_file), int(latent_dim))
                        model_images = build_spectrum_images_for_single_model(bends)
                        latent = encode_spectra(encoder, model_images, batch_size=32)
                        latent = np.asarray(latent, dtype=float).reshape(len(bends), -1)
                        cluster_result = cluster_latent_space(latent, n_clusters=int(n_clusters), random_state=35)
                        cluster_labels = cluster_result.labels
                    metadata["cluster"] = cluster_labels
                    for i in range(min(3, latent.shape[1])):
                        metadata[f"latent_{i + 1}"] = latent[:, i]
                except Exception as exc:
                    st.error(f"Could not run autoencoder clustering: {exc}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

        st.write(f"Detected **{len(bends)}** candidate single bends.")
        st.dataframe(metadata, use_container_width=True)
        if not metadata.empty:
            st.download_button("Download single-bend summary CSV", _to_csv_download(metadata), "single_bend_summary.csv", "text/csv")
        if bends:
            bend_id = st.slider("Bend ID", 0, len(bends) - 1, 0, key="single_bend_id")
            bend = bends[bend_id]
            log_energy = st.checkbox("Show logarithmic energy scale", value=False, key="single_log_energy")
            cwt_result = cwt_energy_from_geometry(bend.x, bend.y, target_points=201, log_energy=log_energy)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_centerline_with_bend(x, y, bend), clear_figure=True)
                st.pyplot(plot_normalized_bend(bend), clear_figure=True)
            with col2:
                st.pyplot(plot_curvature_and_spectrum(cwt_result), clear_figure=True)
                if latent is not None and cluster_labels is not None:
                    st.pyplot(plot_single_latent_space(latent, cluster_labels, bend_id), clear_figure=True)

with compound_tab:
    st.subheader("Compound-bend workflow")
    st.caption("Reach-scale CWT-energy segmentation into simple or compound meander units.")
    if x is None:
        st.info("Upload a centreline file from the sidebar to use this tab.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            points_per_width = st.number_input("Points per width", min_value=2, max_value=100, value=25, step=1)
        with col_b:
            meander_window_widths = st.number_input("CWT window / width", min_value=4.0, max_value=80.0, value=22.0, step=1.0)
        with col_c:
            min_unit_widths = st.number_input("Minimum unit length / width", min_value=0.0, max_value=80.0, value=8.0, step=1.0)
        with col_d:
            valley_prominence = st.number_input("Valley prominence", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

        try:
            units, segmentation = extract_compound_bends(
                x,
                y,
                width=width_source,
                points_per_width=int(points_per_width),
                meander_window_widths=float(meander_window_widths),
                min_unit_widths=float(min_unit_widths),
                valley_prominence=float(valley_prominence),
            )
            unit_table = pd.DataFrame([unit.metadata() for unit in units])
            st.write(
                f"Detected **{len(units)}** CWT-energy meander units; "
                f"**{sum(unit.is_compound for unit in units)}** are compound/complex."
            )
            st.dataframe(unit_table, use_container_width=True)
            if not unit_table.empty:
                st.download_button("Download compound unit summary CSV", _to_csv_download(unit_table), "compound_bend_summary.csv", "text/csv")

            if units:
                selected_unit = st.slider("Compound unit ID", 0, len(units) - 1, 0, key="compound_unit_id")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_compound_units_on_centerline(x, y, units, selected_unit), clear_figure=True)
                with col2:
                    st.pyplot(plot_compound_segmentation_signal(segmentation, units, selected_unit), clear_figure=True)

                spectra = build_spectrum_images_for_compound_model(units)
                st.download_button("Download compound spectra NPY", _to_npy_download(spectra), "compound_spectra.npy", "application/octet-stream")
                st.session_state["compound_units"] = units
                st.session_state["compound_unit_table"] = unit_table
                st.session_state["compound_spectra"] = spectra
                st.session_state["compound_segmentation"] = segmentation
        except Exception as exc:
            st.error(f"Could not run compound segmentation: {exc}")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

with latent_tab:
    st.subheader("Compound latent space")
    st.caption("Encode compound spectra with the separately archived model files.")
    spectra = st.session_state.get("compound_spectra")
    unit_table = st.session_state.get("compound_unit_table")
    if spectra is None or unit_table is None:
        st.info("Run the compound-bend workflow tab first to create compound spectra in memory.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            compound_model_path = st.text_input("Compound model path", value="models/compound_autoencoder.h5")
            model_is_encoder = st.checkbox("Model is encoder-only", value=False)
            latent_layer_name = st.text_input("Latent layer name for full autoencoder", value="Latent_Space", disabled=model_is_encoder)
        with col_b:
            compound_latent_dim = st.number_input("Compound latent dimension", min_value=1, max_value=16, value=2, step=1)
            batch_size = st.number_input("Batch size", min_value=1, max_value=512, value=128, step=16)
            background_path = st.text_input("Background latent cloud path", value="models/world_latent_cloud.npy")

        if st.button("Encode compound units"):
            model_file = Path(compound_model_path)
            if not model_file.exists():
                st.error(f"Model file not found: {model_file}")
            else:
                try:
                    with st.spinner("Encoding compound units..."):
                        encoder = load_cached_compound_encoder(
                            str(model_file),
                            model_is_encoder=bool(model_is_encoder),
                            latent_layer_name=str(latent_layer_name),
                            latent_dim=int(compound_latent_dim),
                        )
                        latent = encode_spectra_with_encoder(encoder, spectra, batch_size=int(batch_size))
                        latent_table = unit_table.merge(
                            pd.DataFrame(
                                {f"latent_{i + 1}": latent[:, i] for i in range(latent.shape[1])}
                                | {"unit_id": np.arange(latent.shape[0], dtype=int)}
                            ),
                            on="unit_id",
                            how="left",
                            validate="one_to_one",
                        )
                        background_file = Path(background_path) if background_path else None
                        background_arg = background_file if background_file and background_file.exists() else None
                        diagnostic_table = build_latent_diagnostics(latent_table, background_latent_path=background_arg)
                        st.session_state["compound_latent_table"] = diagnostic_table
                        st.session_state["compound_background_path"] = str(background_arg) if background_arg else ""
                    st.success(f"Encoded {latent.shape[0]} compound units into a {latent.shape[1]}-D latent space.")
                except Exception as exc:
                    st.error(f"Could not encode compound units: {exc}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

        latent_table = st.session_state.get("compound_latent_table")
        if latent_table is not None:
            st.dataframe(latent_table, use_container_width=True)
            st.download_button("Download compound latent diagnostics CSV", _to_csv_download(latent_table), "compound_latent_diagnostics.csv", "text/csv")
            background = None
            background_state = st.session_state.get("compound_background_path", "")
            if background_state:
                try:
                    background = np.load(Path(background_state))
                except Exception:
                    background = None
            selected = st.slider("Highlight latent unit", 0, len(latent_table) - 1, 0, key="latent_selected_unit")
            st.pyplot(plot_compound_latent(latent_table, background_latent=background, selected_id=selected), clear_figure=True)

with reproducibility_tab:
    st.subheader("Reproducibility and Zenodo files")
    st.markdown(
        """
        This repository is intended to be archived as the **software/GUI Zenodo record**.
        The trained compound autoencoder, extracted encoder and world latent cloud should be
        archived separately in the **model/data Zenodo record** and downloaded locally into `models/`.

        Recommended local model files:

        ```text
        models/compound_autoencoder.h5
        models/world_latent_cloud.npy
        ```

        The full autoencoder is used as the default GUI model because it is the most
        robust option across the tested Windows/Anaconda setup. If an exported
        encoder-only model is available and loads correctly in your TensorFlow/Keras
        environment, tick **Model is encoder-only** and provide that encoder path.

        The model files are intentionally ignored by Git and should not be committed to the repository.
        """
    )
