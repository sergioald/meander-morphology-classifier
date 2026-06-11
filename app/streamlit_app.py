from __future__ import annotations

from pathlib import Path
import tempfile
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from meander_morphology.bends import extract_single_bends
from meander_morphology.clustering import cluster_latent_space
from meander_morphology.cwt import cwt_energy_from_geometry, energy_to_image
from meander_morphology.io import read_centerline_table
from meander_morphology.latent import encode_spectra
from meander_morphology.model import build_encoder_from_autoencoder, load_autoencoder


@st.cache_resource(show_spinner=False)
def load_autoencoder_and_encoder(model_path: str, latent_dim: int = 2):
    autoencoder = load_autoencoder(model_path)
    encoder = build_encoder_from_autoencoder(autoencoder, latent_dim=latent_dim)
    return autoencoder, encoder


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


def plot_latent_space(latent: np.ndarray, labels: np.ndarray, selected_id: int):
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


def build_spectrum_images_for_model(bends) -> np.ndarray:
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


st.set_page_config(page_title="Meander Morphology Classifier", layout="wide")
st.title("Meander Morphology Classifier")
st.caption("Curvature-based single-bend extraction and legacy-compatible isolated CWT-spectrum analysis")

uploaded = st.file_uploader("Upload centerline CSV/TXT/DAT", type=["csv", "txt", "dat"])
width = st.number_input("Fallback channel width", min_value=0.0, value=100.0, step=10.0)
min_chord_widths = st.number_input(
    "Minimum bend chord length / width",
    min_value=0.0,
    value=0.0,
    step=0.5,
    help="Use 5--8 for publication-style filtering. Use 0 to keep all consecutive inflection-point candidates.",
)
endpoint_mode = st.selectbox(
    "Endpoint handling",
    options=["auto", "ignore", "include"],
    index=0,
    help=(
        "auto uses a file endpoint only when endpoint curvature is small enough to be a plausible inflection. "
        "ignore uses only interior sign-change inflections. include always uses endpoints and may include partial edge bends."
    ),
)
endpoint_tolerance = st.slider(
    "Auto endpoint curvature tolerance",
    min_value=0.0,
    max_value=0.30,
    value=0.10,
    step=0.01,
    help="Endpoint is accepted in auto mode when |curvature at endpoint| <= tolerance × max(|curvature|).",
)
log_energy = st.checkbox(
    "Show logarithmic energy scale",
    value=False,
    help="This changes only the diagnostic spectrum figure. Model encoding uses the normalized linear-energy image.",
)

st.divider()
use_autoencoder = st.checkbox(
    "Use Zenodo autoencoder for latent-space clustering",
    value=False,
    help="Loads Autoencoder_Meander_Bend.h5, encodes the extracted bends, and applies K-means in the latent space.",
)
model_path = st.text_input(
    "Autoencoder model path",
    value="models/Autoencoder_Meander_Bend.h5",
    disabled=not use_autoencoder,
)
latent_dim = st.number_input(
    "Latent dimension",
    min_value=1,
    max_value=16,
    value=2,
    step=1,
    disabled=not use_autoencoder,
)
n_clusters = st.number_input(
    "Number of latent-space clusters",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    disabled=not use_autoencoder,
)

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    x, y, width_values = read_centerline_table(tmp_path)
    width_source = width_values if width_values is not None else width
    min_filter = None if min_chord_widths <= 0 else float(min_chord_widths)
    bends = extract_single_bends(
        x,
        y,
        width=width_source,
        min_chord_widths=min_filter,
        endpoint_mode=endpoint_mode,
        endpoint_curvature_tolerance=endpoint_tolerance,
    )

    endpoint_count = sum(b.uses_endpoint_boundary for b in bends)
    metadata = pd.DataFrame([bend.metadata() for bend in bends])
    latent = None
    cluster_labels = None

    if use_autoencoder and bends:
        model_file = Path(model_path)
        if not model_file.exists():
            st.warning(
                f"Model file not found: {model_file}. Run `python scripts/download_model.py --output models` "
                "or update the model path."
            )
        elif len(bends) < int(n_clusters):
            st.warning("The number of bends must be at least the requested number of clusters.")
        else:
            try:
                with st.spinner("Encoding bends with the Zenodo autoencoder..."):
                    _, encoder = load_autoencoder_and_encoder(str(model_file), int(latent_dim))
                    model_images = build_spectrum_images_for_model(bends)
                    latent = encode_spectra(encoder, model_images, batch_size=32)
                    latent = np.asarray(latent, dtype=float).reshape(len(bends), -1)
                    cluster_result = cluster_latent_space(
                        latent,
                        n_clusters=int(n_clusters),
                        random_state=35,
                    )
                    cluster_labels = cluster_result.labels
                metadata["cluster"] = cluster_labels
                for i in range(min(3, latent.shape[1])):
                    metadata[f"latent_{i + 1}"] = latent[:, i]
            except Exception as exc:
                st.error(f"Could not run autoencoder clustering: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())
                latent = None
                cluster_labels = None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Centerline")
        st.caption("The selected bend is highlighted below after choosing a Bend ID.")
    with col2:
        st.subheader("Detected single bends")
        st.write(f"Detected **{len(bends)}** candidate single bends.")
        st.caption(
            f"Endpoint-boundary candidates: {endpoint_count}. Interior candidates are bounded by consecutive curvature sign-change inflection points."
        )
        st.dataframe(metadata)

    if bends:
        bend_id = st.slider("Bend ID", 0, len(bends) - 1, 0)
        bend = bends[bend_id]
        cwt_result = cwt_energy_from_geometry(bend.x, bend.y, target_points=201, log_energy=log_energy)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Selected bend on centerline")
            st.pyplot(plot_centerline_with_bend(x, y, bend), clear_figure=True)
            st.subheader("Normalized isolated bend")
            st.pyplot(plot_normalized_bend(bend), clear_figure=True)
        with col4:
            st.subheader("CWT energy spectrum")
            st.caption(
                "The spectrum is computed from the selected normalized single bend. "
                "The bend is reflected only while recomputing curvature, then cropped back to the central bend before CWT. "
                "Adjacent bends are not included."
            )
            st.pyplot(plot_curvature_and_spectrum(cwt_result), clear_figure=True)

            if latent is not None and cluster_labels is not None:
                st.subheader("Autoencoder latent-space clustering")
                st.write(
                    f"Selected bend cluster: **{int(cluster_labels[bend_id])}** "
                    f"from **{int(n_clusters)}** K-means clusters."
                )
                st.pyplot(plot_latent_space(latent, cluster_labels, bend_id), clear_figure=True)
else:
    st.info("Upload a centerline file with x, y and optionally width columns to begin.")
