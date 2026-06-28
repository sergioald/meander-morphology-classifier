from __future__ import annotations


def test_streamlit_app_importable():
    import importlib.util
    from pathlib import Path

    app_path = Path("app/streamlit_app.py")
    assert app_path.exists()
    spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
    assert spec is not None
