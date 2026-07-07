"""Lightweight public Streamlit demo for meander-morphology-classifier."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from meander_morphology.compound import compound_bends_to_metadata_rows, extract_compound_bends


st.set_page_config(
    page_title="Meander Morphology Classifier Demo",
    page_icon="〰️",
    layout="wide",
)


def make_demo_centerline(n: int = 420, width: float = 40.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a synthetic public-safe meander centreline for one-click demo use."""
    t = np.linspace(0.0, 1.0, int(n))
    x = 2400.0 * t
    y = (
        230.0 * np.sin(2.0 * np.pi * (3.25 * t + 0.05))
        + 85.0 * np.sin(2.0 * np.pi * (7.1 * t + 0.17))
        + 35.0 * np.sin(2.0 * np.pi * (12.0 * t + 0.31))
    )
    y += 120.0 * (t - 0.5)
    widths = np.full_like(x, float(width), dtype=float)
    return x, y, widths


def read_uploaded_centerline(uploaded_file, width_column: str = "width") -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read a simple CSV/TXT/DAT centreline from an uploaded file."""
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix == ".csv":
        table = pd.read_csv(BytesIO(raw))
    else:
        table = pd.read_csv(BytesIO(raw), sep=None, engine="python")

    lower_map = {str(col).strip().lower(): col for col in table.columns}
    if "x" in lower_map and "y" in lower_map:
        x = table[lower_map["x"]].to_numpy(dtype=float)
        y = table[lower_map["y"]].to_numpy(dtype=float)
        width = None
        if width_column and width_column.lower() in lower_map:
            width = table[lower_map[width_column.lower()]].to_numpy(dtype=float)
        return x, y, width

    numeric = table.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError("Input must contain at least two numeric columns, or named x/y columns.")
    x = numeric.iloc[:, 0].to_numpy(dtype=float)
    y = numeric.iloc[:, 1].to_numpy(dtype=float)
    width = numeric.iloc[:, 3].to_numpy(dtype=float) if numeric.shape[1] >= 4 else None
    return x, y, width


def unit_summary(units) -> pd.DataFrame:
    """Return compact public-demo metadata for extracted units."""
    if not units:
        return pd.DataFrame()
    table = pd.DataFrame(compound_bends_to_metadata_rows(units))
    preferred = [
        "unit_id",
        "start_index",
        "end_index",
        "width",
        "sinuosity",
        "maturity_index",
        "chord_widths",
        "length_widths",
        "n_internal_inflections",
        "n_lobes",
        "is_compound",
        "boundary_method",
    ]
    existing = [col for col in preferred if col in table.columns]
    return table[existing].copy()


def plot_centerline_units(x: np.ndarray, y: np.ndarray, units, selected_unit: int | None = None):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(x, y, linewidth=1.0, color="0.35", label="centreline")
    for unit in units:
        highlight = selected_unit is not None and unit.unit_id == selected_unit
        ax.plot(
            unit.raw_x,
            unit.raw_y,
            linewidth=3.0 if highlight else 1.6,
            alpha=1.0 if highlight else 0.7,
            label="selected unit" if highlight else None,
        )
        if len(unit.raw_x):
            mid = len(unit.raw_x) // 2
            ax.annotate(
                str(unit.unit_id),
                (unit.raw_x[mid], unit.raw_y[mid]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Detected meander units on centreline")
    if selected_unit is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_segmentation_signal(segmentation, selected_unit=None, units=None):
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.plot(segmentation.s, segmentation.normalised_energy, linewidth=1.5, label="normalised CWT corridor energy")
    for boundary in segmentation.boundary_indices:
        idx = int(boundary)
        ax.axvline(segmentation.s[idx], color="0.45", linewidth=0.8, alpha=0.6)
    if selected_unit is not None and units and 0 <= selected_unit < len(units):
        unit = units[selected_unit]
        ax.axvspan(
            segmentation.s[unit.start_index],
            segmentation.s[unit.end_index],
            alpha=0.18,
            label=f"selected unit {selected_unit}",
        )
    ax.set_xlabel("along-centreline distance")
    ax.set_ylabel("normalised energy")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Compound-unit segmentation signal")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def to_csv_bytes(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False).encode("utf-8")


st.title("Meander Morphology Classifier")
st.caption(
    "One-click public demo for centreline-based compound-bend extraction and "
    "CWT-energy segmentation. Model-backed latent-space workflows are available "
    "in the full local repository."
)

st.info(
    "This hosted demo uses synthetic/example or user-uploaded centreline data and does not download "
    "large Zenodo model files. For the full autoencoder workflow, clone the repository and follow the README."
)

with st.sidebar:
    st.header("Input centreline")
    uploaded = st.file_uploader("Upload CSV/TXT/DAT", type=["csv", "txt", "dat"])
    width_column = st.text_input("Width column name", value="width")
    fallback_width = st.number_input("Fallback channel width", min_value=1.0, value=40.0, step=5.0)
    use_file_width = st.checkbox("Use width values from file when available", value=True)

    st.header("Compound extraction")
    points_per_width = st.slider("Resampling density: points per width", min_value=8, max_value=40, value=18, step=2)
    meander_window_widths = st.slider("CWT smoothing window [widths]", min_value=8.0, max_value=35.0, value=22.0, step=1.0)
    min_unit_widths = st.slider("Minimum unit length [widths]", min_value=3.0, max_value=20.0, value=8.0, step=0.5)
    valley_prominence = st.slider("Valley prominence", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    st.caption("Default data are synthetic and public-safe.")

try:
    if uploaded is None:
        x, y, width_values = make_demo_centerline(width=fallback_width)
        source_label = "Synthetic demo centreline"
    else:
        x, y, width_values = read_uploaded_centerline(uploaded, width_column=width_column.strip() or "width")
        source_label = f"Uploaded file: {uploaded.name}"

    if width_values is not None and use_file_width:
        width_for_workflow = width_values
        width_display = float(np.nanmean(width_values))
    else:
        width_for_workflow = float(fallback_width)
        width_display = float(fallback_width)

    st.subheader("Input summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Source", source_label)
    c2.metric("Centreline points", f"{len(x)}")
    c3.metric("Representative width", f"{width_display:.3g}")

    with st.spinner("Extracting compound meander units..."):
        units, segmentation = extract_compound_bends(
            x,
            y,
            width=width_for_workflow,
            points_per_width=int(points_per_width),
            meander_window_widths=float(meander_window_widths),
            min_unit_widths=float(min_unit_widths),
            valley_prominence=float(valley_prominence),
        )

    summary = unit_summary(units)
    selected_unit = None
    if len(units):
        selected_unit = st.slider("Highlight unit", min_value=0, max_value=len(units) - 1, value=0, step=1)

    st.subheader("Detected meander units")
    m1, m2, m3 = st.columns(3)
    m1.metric("Detected units", f"{len(units)}")
    m2.metric("Compound units", f"{int(summary['is_compound'].sum()) if 'is_compound' in summary else 0}")
    m3.metric("Mean sinuosity", f"{summary['sinuosity'].mean():.3f}" if "sinuosity" in summary and not summary.empty else "n/a")

    st.pyplot(plot_centerline_units(x, y, units, selected_unit=selected_unit), clear_figure=True)
    st.pyplot(plot_segmentation_signal(segmentation, selected_unit=selected_unit, units=units), clear_figure=True)

    st.subheader("Unit summary")
    st.dataframe(summary, use_container_width=True)
    st.download_button(
        "Download unit summary CSV",
        data=to_csv_bytes(summary),
        file_name="compound_unit_summary.csv",
        mime="text/csv",
        disabled=summary.empty,
    )

    with st.expander("What this hosted demo does and does not show"):
        st.markdown(
            """
            **Shown here**

            - public-safe centreline loading;
            - compound-unit extraction from curvature/CWT-energy structure;
            - segmentation diagnostics;
            - downloadable unit summary outputs.

            **Not shown here**

            - TensorFlow/Keras model loading;
            - Zenodo model-file downloads;
            - full paper-figure reproduction;
            - exhaustive validation on the original research datasets.

            The model-backed workflows remain available in the local GUI and command-line scripts.
            """
        )

except Exception as exc:
    st.error(f"Demo failed: {exc}")
    st.exception(exc)
