from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from meander_morphology.bends import extract_single_bends
from meander_morphology.cwt import spectrum_image
from meander_morphology.io import read_centerline_table

st.set_page_config(page_title="Meander Morphology Classifier", layout="wide")
st.title("Meander Morphology Classifier")
st.caption("Curvature-based single-bend extraction and CWT-spectrum analysis")

uploaded = st.file_uploader("Upload centerline CSV/TXT/DAT", type=["csv", "txt", "dat"])
width = st.number_input("Fallback channel width", min_value=0.0, value=100.0, step=10.0)

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    x, y, width_values = read_centerline_table(tmp_path)
    width_source = width_values if width_values is not None else width
    bends = extract_single_bends(x, y, width=width_source)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Centerline")
        st.line_chart(pd.DataFrame({"x": x, "y": y}).set_index("x"))
    with col2:
        st.subheader("Detected bends")
        st.write(f"Detected **{len(bends)}** candidate single bends.")
        st.dataframe(pd.DataFrame([bend.metadata() for bend in bends]))

    if bends:
        bend_id = st.slider("Bend ID", 0, len(bends) - 1, 0)
        bend = bends[bend_id]
        image = spectrum_image(bend.curvature)
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Normalized bend")
            st.line_chart(pd.DataFrame({"x": bend.x, "y": bend.y}).set_index("x"))
        with col4:
            st.subheader("CWT energy spectrum")
            st.image((image * 255).astype(np.uint8), clamp=True)
else:
    st.info("Upload a centerline file with x, y and optionally width columns to begin.")
