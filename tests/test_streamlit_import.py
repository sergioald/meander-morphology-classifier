from __future__ import annotations

from pathlib import Path


def test_streamlit_app_has_no_deprecated_container_width_argument():
    text = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "use_container_width" not in text


def test_streamlit_app_compiles():
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    compile(source, "app/streamlit_app.py", "exec")


def test_gui_labels_exact_and_enhanced_model_cwt_images():
    text = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    assert "Exact model input" in text
    assert "Enhanced preview" in text
    assert "detected meander units" in text
