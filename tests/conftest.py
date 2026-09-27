"""Keep every default runtime created by tests away from local application data."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from local_rag_backend.settings import load_settings_from_yaml

_runtime_directory = TemporaryDirectory(prefix="rag-tests-")
_root = Path(_runtime_directory.name)
_configuration = load_settings_from_yaml(Path(__file__).resolve().parents[1] / "config.yaml")
_payload = _configuration.model_dump(mode="json")
_payload.update(
    data_dir=str(_root / "data"),
    sqlite_url=f"sqlite:///{_root / 'app.db'}",
    index_path=str(_root / "index.faiss"),
    id_map_path=str(_root / "index.ids.json"),
    embedding_cache_db_path=str(_root / "embeddings.sqlite"),
    perf_metrics_out_path=None,
    lock_metrics_path=None,
)
_config_path = _root / "config.yaml"
_config_path.write_text(yaml.safe_dump(_payload), encoding="utf-8")
os.environ["RAG_CONFIG_PATH"] = str(_config_path)


@pytest.fixture(autouse=True)
def isolated_runtime_paths(tmp_path, monkeypatch):
    from local_rag_backend.composition.factory import reset_app_context
    from local_rag_backend.settings import get_settings

    reset_app_context()
    settings_obj = get_settings()
    for field, value in {
        "data_dir": tmp_path / "data",
        "sqlite_url": f"sqlite:///{tmp_path / 'app.db'}",
        "index_path": str(tmp_path / "index.faiss"),
        "id_map_path": str(tmp_path / "index.ids.json"),
        "embedding_cache_db_path": tmp_path / "embeddings.sqlite",
    }.items():
        monkeypatch.setattr(settings_obj, field, value)
    yield
    reset_app_context()
