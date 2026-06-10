from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from meander_morphology.bends import extract_single_bends
from meander_morphology.cwt import spectrum_image
from meander_morphology.io import read_centerline_table


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


st.set_page_config(page_title="Meander Morphology Classifier", layout="wide")
st.title("Meander Morphology Classifier")
st.caption("Curvature-based single-bend extraction and isolated CWT-spectrum analysis")

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
use_cwt_padding = st.checkbox(
    "Mirror-pad bend before CWT, then crop back to selected bend",
    value=True,
    help="This reduces edge artefacts without including adjacent bends in the saved spectrum.",
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
        st.dataframe(pd.DataFrame([bend.metadata() for bend in bends]))

    if bends:
        bend_id = st.slider("Bend ID", 0, len(bends) - 1, 0)
        bend = bends[bend_id]
        image = spectrum_image(bend.curvature, pad=use_cwt_padding)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Selected bend on centerline")
            st.pyplot(plot_centerline_with_bend(x, y, bend), clear_figure=True)
            st.subheader("Normalized isolated bend")
            st.pyplot(plot_normalized_bend(bend), clear_figure=True)
        with col4:
            st.subheader("CWT energy spectrum")
            st.caption(
                "Spectrum is computed from the selected isolated bend curvature. "
                "Mirror padding is cropped back to the selected bend and does not add adjacent bends."
            )
            st.image((image * 255).astype(np.uint8), clamp=True)
else:
    st.info("Upload a centerline file with x, y and optionally width columns to begin.")
