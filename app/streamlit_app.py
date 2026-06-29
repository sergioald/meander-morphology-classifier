from __future__ import annotations

from io import BytesIO
import hashlib
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
from meander_morphology.model import (
    ZENODO_MODEL_FILENAME,
    ZENODO_MODEL_MD5,
    ZENODO_MODEL_URL,
    build_encoder_from_autoencoder,
    load_autoencoder,
)


APP_ROOT = Path(__file__).resolve().parents[1]


def resolve_local_path(path_text: str | Path | None) -> Path:
    """Resolve GUI paths relative to the repository root when they are not absolute."""
    path = Path(path_text or "")
    if path.is_absolute():
        return path
    return APP_ROOT / path



def md5sum(path: Path) -> str:
    """Return the MD5 checksum of a local file."""
    h = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_single_autoencoder_from_zenodo(target: Path, *, force: bool = False) -> tuple[Path, str]:
    """Download the public single-bend autoencoder from Zenodo into ``target``."""
    import requests

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target, md5sum(target)

    tmp_target = target.with_suffix(target.suffix + ".part")
    with requests.get(ZENODO_MODEL_URL, stream=True, timeout=90) as response:
        response.raise_for_status()
        with tmp_target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_target.replace(target)

    digest = md5sum(target)
    if digest != ZENODO_MODEL_MD5:
        raise RuntimeError(f"MD5 mismatch for {target.name}: got {digest}, expected {ZENODO_MODEL_MD5}")
    return target, digest


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


def read_uploaded_centerline(
    uploaded_file,
    *,
    width_column: str | None = "width",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    return read_centerline_table(tmp_path, width_column=width_column)


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
    """Plot single-bend latent clustering with a clear selected-bend marker."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    labels = np.asarray(labels)
    if latent.shape[1] >= 2:
        scatter = ax.scatter(latent[:, 0], latent[:, 1], c=labels, s=34, alpha=0.85)
        ax.scatter(
            latent[selected_id, 0],
            latent[selected_id, 1],
            marker="x",
            s=95,
            linewidths=2.0,
            label="selected bend",
        )
        ax.set_xlabel("latent 1", fontsize=8)
        ax.set_ylabel("latent 2", fontsize=8)
        fig.colorbar(scatter, ax=ax, label="cluster", shrink=0.78)
    else:
        ax.scatter(np.arange(latent.shape[0]), latent[:, 0], c=labels, s=34, alpha=0.85)
        ax.scatter([selected_id], [latent[selected_id, 0]], marker="x", s=95, linewidths=2.0, label="selected bend")
        ax.set_xlabel("bend ID", fontsize=8)
        ax.set_ylabel("latent value", fontsize=8)
    ax.set_title("Single-bend latent-space clusters", fontsize=10)
    ax.legend(loc="best", fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig

def plot_compound_units_on_centerline(x: np.ndarray, y: np.ndarray, units, selected_id: int | None = None):
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.plot(x, y, linewidth=1.0, label="centerline")
    for unit in units:
        linewidth = 2.2 if unit.unit_id == selected_id else 1.2
        alpha = 1.0 if unit.unit_id == selected_id else 0.55
        ax.plot(unit.raw_x, unit.raw_y, linewidth=linewidth, alpha=alpha)
        ax.scatter([unit.raw_x[0], unit.raw_x[-1]], [unit.raw_y[0], unit.raw_y[-1]], s=12, alpha=alpha)
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
    fig, ax = plt.subplots(figsize=(5.8, 2.6))
    ax.plot(segmentation.s, segmentation.normalised_energy, linewidth=1.5, label="normalised corridor energy")
    for boundary in segmentation.boundary_indices:
        ax.axvline(segmentation.s[int(boundary)], linewidth=0.8, alpha=0.45)
    if units and selected_id is not None:
        unit = units[selected_id]
        ax.axvspan(segmentation.s[unit.start_index], segmentation.s[unit.end_index], alpha=0.12)
    ax.set_xlabel("along-centreline distance")
    ax.set_ylabel("normalised energy")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig

def plot_compound_reach_cwt(
    segmentation,
    x: np.ndarray,
    y: np.ndarray,
    units=None,
    selected_id: int | None = None,
    *,
    max_display_frequency: float | None = 0.004):
    """Compact four-panel reach diagnostic following the legacy research-plot layout."""
    energy = np.asarray(segmentation.energy, dtype=float)
    freqs = np.asarray(segmentation.frequencies, dtype=float)
    s = np.asarray(segmentation.s, dtype=float)
    ridge = np.asarray(segmentation.ridge_indices, dtype=int)
    trough = np.asarray(segmentation.trough_indices, dtype=int)

    finite_energy = np.where(np.isfinite(energy), energy, 0.0)
    display_energy = np.log1p(np.maximum(finite_energy, 0.0))

    valid = np.concatenate([ridge, trough]) if ridge.size and trough.size else np.arange(len(freqs))
    valid = valid[(valid >= 0) & (valid < len(freqs))]
    if valid.size:
        lo = max(0, int(np.nanmin(valid)) - 4)
        hi = min(len(freqs) - 1, int(np.nanmax(valid)) + 8)
    else:
        lo, hi = 0, len(freqs) - 1
    if max_display_frequency is not None and max_display_frequency > 0:
        freq_limit_idx = np.where(freqs <= float(max_display_frequency))[0]
        if freq_limit_idx.size:
            lo = min(lo, int(freq_limit_idx[0]))
            hi = max(hi, int(freq_limit_idx[-1]))
    if hi <= lo:
        lo, hi = 0, len(freqs) - 1
    if max_display_frequency is not None and float(max_display_frequency) > 0:
        display_candidates = np.where(freqs <= float(max_display_frequency))[0]
        if display_candidates.size:
            lo = min(lo, int(display_candidates[0]))
            hi = max(hi, int(display_candidates[-1]))

    freq_slice = slice(lo, hi + 1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4,
        1,
        figsize=(6.2, 5.6),
        height_ratios=[1.05, 0.8, 0.8, 2.0],
        sharex=False,
    )
    plt.subplots_adjust(hspace=0.15)

    ax1.pcolormesh(s, freqs[freq_slice], display_energy[freq_slice, :], shading="auto", cmap="jet")
    if ridge.size == s.size:
        ax1.plot(s, freqs[np.clip(ridge, 0, len(freqs) - 1)], linewidth=1.1, label="ridge")
    if trough.size == s.size:
        ax1.plot(s, freqs[np.clip(trough, 0, len(freqs) - 1)], linewidth=1.1, label="trough")
    for boundary in segmentation.boundary_indices:
        ax1.axvline(segmentation.s[int(boundary)], linewidth=0.8, alpha=0.85)
    ax1.set_ylabel("freq [1/unit]", fontsize=8)
    ax1.set_title("Reach-scale CWT diagnostic", fontsize=10)
    ax1.tick_params(labelsize=7)
    if max_display_frequency is not None and max_display_frequency > 0:
        ax1.set_ylim(freqs[freq_slice].min(), max(freqs[freq_slice].max(), float(max_display_frequency)))

    ax2.plot(s, segmentation.normalised_energy, linewidth=1.1)
    for boundary in segmentation.boundary_indices:
        ax2.plot(segmentation.s[int(boundary)], segmentation.normalised_energy[int(boundary)], ".", ms=6)
    ax2.axhline(y=0.15, linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("ΣE", fontsize=8)
    ax2.set_ylim(-0.04, 1.04)
    ax2.tick_params(labelsize=7)

    curvature = np.full_like(s, np.nan, dtype=float)
    if units:
        for unit in units:
            start = int(unit.start_index)
            end = int(unit.end_index)
            vals = np.asarray(unit.curvature, dtype=float)
            if vals.size and end > start:
                target = np.linspace(start, end, vals.size)
                idx = np.arange(start, end + 1)
                curvature[start : end + 1] = np.interp(idx, target, vals)
    if not np.isfinite(curvature).any():
        curvature = np.gradient(segmentation.normalised_energy, edge_order=1)
    ax3.plot(s, curvature, linewidth=0.9)
    for boundary in segmentation.boundary_indices:
        ax3.plot(segmentation.s[int(boundary)], curvature[int(boundary)], ".", ms=5)
    ax3.axhline(y=0.0, linestyle="--", linewidth=0.7, alpha=0.55)
    ax3.set_ylabel("curv", fontsize=8)
    ax3.set_xlabel("s", fontsize=8)
    ax3.tick_params(labelsize=7)

    ax4.plot(x, y, linewidth=0.9)
    if units:
        for unit in units:
            alpha = 1.0 if unit.unit_id == selected_id else 0.42
            lw = 1.8 if unit.unit_id == selected_id else 0.75
            ax4.plot(unit.raw_x, unit.raw_y, linewidth=lw, alpha=alpha)
            ax4.scatter([unit.raw_x[0], unit.raw_x[-1]], [unit.raw_y[0], unit.raw_y[-1]], s=14, alpha=alpha)
            if len(unit.raw_x):
                mid = len(unit.raw_x) // 2
                ax4.annotate(str(unit.unit_id), (unit.raw_x[mid], unit.raw_y[mid]), fontsize=6)
    ax4.axis("equal")
    ax4.set_xlabel("x", fontsize=8)
    ax4.set_ylabel("y", fontsize=8)
    ax4.tick_params(labelsize=7)

    fig.tight_layout(pad=0.55)
    return fig


def _contrast_stretch_image(image: np.ndarray, *, low: float = 1.0, high: float = 99.5, gamma: float = 0.65) -> np.ndarray:
    """Return a display-only contrast-stretched copy of a 64 x 64 spectrum image."""
    arr = np.asarray(image, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    lo = float(np.nanpercentile(arr[finite], low))
    hi = float(np.nanpercentile(arr[finite], high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(arr, 0.0, 1.0)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return np.power(out, gamma)


def plot_compound_spectrum_image(image: np.ndarray, unit_id: int | None = None, *, enhanced: bool = True):
    """Plot the exact 64 x 64 rasterised compound-autoencoder input.

    This view intentionally has no physical axes because the training images were
    saved as axis-free contour PNGs before being passed to the autoencoder.
    """
    arr = np.asarray(image, dtype=float)
    shown = _contrast_stretch_image(arr) if enhanced else arr
    fig, ax = plt.subplots(figsize=(2.0, 2.0))
    ax.imshow(shown, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", aspect="equal")
    title = "Exact 64 x 64 autoencoder input" if unit_id is None else f"Exact 64 x 64 autoencoder input: unit {unit_id}"
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    fig.tight_layout(pad=0.25)
    return fig


def plot_compound_training_preview(preview_bundle: dict[str, np.ndarray], unit_id: int | None = None):
    """Plot the human-readable compound CWT using the research-script convention."""
    cwt_matrix = np.asarray(preview_bundle["cwt_matrix"], dtype=float)
    s_axis = np.asarray(preview_bundle["s_axis"], dtype=float)
    l_axis = np.asarray(preview_bundle["l_axis"], dtype=float)

    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    finite = np.isfinite(cwt_matrix)
    vmin = float(np.nanmean(cwt_matrix[finite]) + np.nanstd(cwt_matrix[finite])) if finite.any() else None

    ax.contourf(
        s_axis,
        l_axis,
        cwt_matrix,
        levels=8,
        extend="both",
        cmap="binary",
        vmin=vmin,
    )
    title = "Training-style compound CWT" if unit_id is None else f"Training-style compound CWT: unit {unit_id}"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$S_{bend} / S_{bend,max}$", fontsize=7)
    ax.set_ylabel("l = 1 / frequency", fontsize=7)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.45)
    return fig


def plot_compound_latent(
    table: pd.DataFrame,
    background_latent: np.ndarray | None = None,
    selected_id: int | None = None,
    *,
    show_unit_labels: bool = True,
):
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    if background_latent is not None and background_latent.ndim == 2 and background_latent.shape[1] >= 2:
        ax.scatter(
            background_latent[:, 0],
            background_latent[:, 1],
            s=1.2,
            alpha=0.08,
            label="reference meander cloud",
        )
    if {"latent_1", "latent_2"}.issubset(table.columns):
        ax.scatter(table["latent_1"], table["latent_2"], s=28, label="detected meander units")
        if show_unit_labels:
            for _, row in table.iterrows():
                ax.annotate(
                    str(int(row.get("unit_id", 0))),
                    (row["latent_1"], row["latent_2"]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        if selected_id is not None and selected_id < len(table):
            row = table.iloc[selected_id]
            ax.scatter([row["latent_1"]], [row["latent_2"]], marker="x", s=75, linewidths=2, label="selected unit")
    ax.set_xlabel("latent_1", fontsize=8)
    ax.set_ylabel("latent_2", fontsize=8)
    ax.set_title("Compound-bend latent space", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout(pad=0.5)
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
    """Build compound images with the same polarity/style as the trained compound model."""
    from meander_morphology.cwt import legacy_compound_training_image_from_curvature

    images = []
    for unit in units:
        images.append(legacy_compound_training_image_from_curvature(unit.curvature, image_size=image_size))
    return np.asarray(images, dtype="float32")

def build_compound_training_preview(unit, *, image_size: int = 64) -> dict[str, np.ndarray]:
    """Return raw training-style CWT and exact model image for one compound unit."""
    from meander_morphology.cwt import legacy_compound_training_preview_from_curvature

    return legacy_compound_training_preview_from_curvature(unit.curvature, image_size=image_size)



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
    width_column_name = st.text_input(
        "Width column name",
        value="width",
        help="For CSV files with headers. Leave blank to ignore width columns. Headerless TXT/DAT files use the 4th numeric column as width when present.",
    )
    use_file_width = st.checkbox("Use width values from file when available", value=True)
    st.caption("Preferred CSV columns: x, y and optionally width. Headerless files may use x, y, z, width.")

x = y = width_values = None
width_source = fallback_width
if uploaded is not None:
    try:
        width_column = width_column_name.strip() or None
        x, y, width_values = read_uploaded_centerline(uploaded, width_column=width_column)
        if width_values is not None and use_file_width:
            width_source = width_values
        else:
            width_source = fallback_width
        with st.sidebar:
            st.success(f"Loaded {len(x)} centreline points")
            if width_values is not None:
                st.write(
                    "Width detected: "
                    f"mean={float(np.nanmean(width_values)):.3g}, "
                    f"min={float(np.nanmin(width_values)):.3g}, "
                    f"max={float(np.nanmax(width_values)):.3g}"
                )
                if use_file_width:
                    st.caption("Using file width values for extraction and normalisation.")
                else:
                    st.caption("Width values were found but the fallback constant width is being used.")
            else:
                st.info("No width values detected; using fallback channel width.")
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

        st.markdown("**Optional single-bend latent-space clustering**")
        use_autoencoder = st.checkbox("Use single-bend Zenodo autoencoder", value=False, key="single_cluster_settings")
        model_path = st.text_input(
            "Single-bend autoencoder path",
            value=f"models/{ZENODO_MODEL_FILENAME}",
            disabled=not use_autoencoder,
        )
        if use_autoencoder:
            model_file_for_download = resolve_local_path(model_path)
            st.caption(
                "Public single-bend model source: Zenodo record 10.5281/zenodo.13913710. "
                "The model file is ignored by Git and should stay local."
            )
            dl_col, force_col, status_col = st.columns([0.25, 0.25, 0.50])
            with force_col:
                force_single_download = st.checkbox("Overwrite local file", value=False, key="force_single_model_download")
            with dl_col:
                if st.button("Download single-bend model", key="download_single_model"):
                    try:
                        with st.spinner("Downloading single-bend autoencoder from Zenodo..."):
                            saved_path, digest = download_single_autoencoder_from_zenodo(model_file_for_download, force=force_single_download)
                        st.success(f"Downloaded {saved_path.name} to {saved_path.parent}. MD5: {digest}")
                    except Exception as exc:
                        st.error(f"Could not download the Zenodo single-bend model: {exc}")
                        with st.expander("Technical details"):
                            st.code(traceback.format_exc())
            with status_col:
                if model_file_for_download.exists():
                    st.info(f"Local model found: {model_file_for_download}")
                else:
                    st.warning(f"Local model not found yet: {model_file_for_download}")
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
            model_file = resolve_local_path(model_path)
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
        st.dataframe(metadata, width="stretch")
        if not metadata.empty:
            st.download_button("Download single-bend summary CSV", _to_csv_download(metadata), "single_bend_summary.csv", "text/csv")
        if bends:
            bend_id = st.slider("Bend ID", 0, len(bends) - 1, 0, key="single_bend_id")
            bend = bends[bend_id]
            log_energy = st.checkbox("Show logarithmic energy scale", value=False, key="single_log_energy")
            cwt_result = cwt_energy_from_geometry(bend.x, bend.y, target_points=201, log_energy=log_energy)
            if latent is not None and cluster_labels is not None:
                st.subheader("Single-bend latent-space clustering")
                st.write(
                    f"Selected bend cluster: **{int(cluster_labels[bend_id])}** "
                    f"from **{int(n_clusters)}** K-means clusters."
                )
                cluster_summary = (
                    pd.Series(cluster_labels, name="cluster")
                    .value_counts()
                    .sort_index()
                    .rename_axis("cluster")
                    .reset_index(name="n_bends")
                )
                lat_col, sum_col, _ = st.columns([0.42, 0.22, 0.36])
                with lat_col:
                    st.pyplot(plot_single_latent_space(latent, cluster_labels, bend_id), clear_figure=True)
                with sum_col:
                    st.dataframe(cluster_summary, width="content")
            elif use_autoencoder:
                st.info("No latent-space plot is available yet. Check that the single-bend model path exists and that the number of bends is at least the requested number of clusters.")

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_centerline_with_bend(x, y, bend), clear_figure=True)
                st.pyplot(plot_normalized_bend(bend), clear_figure=True)
            with col2:
                st.pyplot(plot_curvature_and_spectrum(cwt_result), clear_figure=True)

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
        show_reach_cwt = st.checkbox("Show reach-scale CWT energy", value=True)
        reach_cwt_max_frequency = st.number_input(
            "Reach CWT display max frequency",
            min_value=0.0005,
            max_value=0.02,
            value=0.004,
            step=0.0005,
            format="%.4f",
            help="Display-only upper frequency limit for the reach-scale diagnostic. The segmentation calculation is unchanged.",
        )
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
            st.dataframe(unit_table, width="stretch")
            if not unit_table.empty:
                st.download_button("Download compound unit summary CSV", _to_csv_download(unit_table), "compound_bend_summary.csv", "text/csv")

            if units:
                selected_unit = st.slider("Compound unit ID", 0, len(units) - 1, 0, key="compound_unit_id")
                spectra = build_spectrum_images_for_compound_model(units)
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_compound_units_on_centerline(x, y, units, selected_unit), clear_figure=True)
                with col2:
                    st.pyplot(plot_compound_segmentation_signal(segmentation, units, selected_unit), clear_figure=True)

                if show_reach_cwt:
                    diag_col, _ = st.columns([0.72, 0.28])
                    with diag_col:
                        st.pyplot(
                            plot_compound_reach_cwt(segmentation, x, y, units, selected_unit, max_display_frequency=float(reach_cwt_max_frequency)),
                            clear_figure=True,
                        )
                    st.caption(
                        "Legacy-style reach diagnostic: reach CWT, corridor-energy signal, curvature panel and centreline boundaries. "
                        "This is different from the 64 x 64 model image saved for each extracted unit."
                    )

                img_exact_col, img_preview_col, _ = st.columns([0.23, 0.23, 0.54])
                st.markdown("**Selected-unit CWT image used by the compound model**")
                st.caption(
                    "The default plot now follows the original compound-training script: "
                    "CWT is shown with horizontal axis $S_{bend} / S_{bend,max}$ and vertical axis $l = 1 / frequency$. "
                    "The exact 64 x 64 model input is available below only for reproducibility."
                )
                preview_bundle = build_compound_training_preview(units[selected_unit])
                preview_col, _ = st.columns([0.42, 0.58])
                with preview_col:
                    st.pyplot(plot_compound_training_preview(preview_bundle, selected_unit), clear_figure=True)

                show_exact_model_input = st.checkbox("Show exact 64 x 64 model input array", value=False)
                if show_exact_model_input:
                    exact_col, _ = st.columns([0.30, 0.70])
                    with exact_col:
                        st.pyplot(plot_compound_spectrum_image(preview_bundle["model_image"], selected_unit, enhanced=False), clear_figure=True)
                st.caption(
                    "The training-style plot is for interpretation and uses the physical/legacy period axis. "
                    "The hidden exact 64 x 64 array is the resized autoencoder input, so it is the only view that should use pixel-like axes."
                )
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
            background_path = st.text_input("Background latent cloud path", value="models/world_latent_cloud.npy")
            st.caption("Inference uses the trained 2-D latent space and a fixed batch size of 128. Training from zero is not exposed in the GUI.")
        compound_latent_dim = 2
        batch_size = 128

        if st.button("Encode compound units"):
            model_file = resolve_local_path(compound_model_path)
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
                        background_file = resolve_local_path(background_path) if background_path else None
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
            st.dataframe(latent_table, width="stretch")
            st.download_button("Download compound latent diagnostics CSV", _to_csv_download(latent_table), "compound_latent_diagnostics.csv", "text/csv")
            background = None
            background_state = st.session_state.get("compound_background_path", "")
            if background_state:
                try:
                    background = np.load(resolve_local_path(background_state))
                except Exception:
                    background = None
            selected = st.slider("Highlight latent unit", 0, len(latent_table) - 1, 0, key="latent_selected_unit")
            show_compound_latent_labels = st.checkbox("Show unit numbers in latent-space plot", value=True)
            lat_col, _ = st.columns([0.56, 0.44])
            with lat_col:
                st.pyplot(plot_compound_latent(latent_table, background_latent=background, selected_id=selected, show_unit_labels=show_compound_latent_labels), clear_figure=True)

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

        Width handling:

        ```text
        CSV with headers: x, y, width
        Headerless TXT/DAT: first column = x, second column = y, fourth column = width
        ```

        The full autoencoder is used as the default GUI model because it is the most
        robust option across the tested Windows/Anaconda setup. If an exported
        encoder-only model is available and loads correctly in your TensorFlow/Keras
        environment, tick **Model is encoder-only** and provide that encoder path.

        The model files are intentionally ignored by Git and should not be committed to the repository.
        """
    )
