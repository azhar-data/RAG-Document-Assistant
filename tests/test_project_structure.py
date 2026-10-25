from pathlib import Path


def test_core_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "README.md").exists()
    assert (root / "rag_app" / "pipeline.py").exists()
