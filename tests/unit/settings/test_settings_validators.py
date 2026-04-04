# tests/unit/settings/test_settings_validators.py
from pathlib import Path

import pytest

from local_rag_backend.settings import Settings, load_settings_from_yaml


def test_sqlite_url_validator():
    with pytest.raises(ValueError):
        Settings(sqlite_url="postgres://x")


def test_elasticsearch_backend_requires_es_base_url():
    with pytest.raises(ValueError, match="es_base_url is required"):
        Settings(
            persistence_backend="elasticsearch",
            retrieval_mode="dense",
            es_base_url=None,
        )


def test_elasticsearch_backend_rejects_sparse():
    with pytest.raises(ValueError, match="supports retrieval_mode=sparse only when"):
        Settings(
            persistence_backend="elasticsearch",
            retrieval_mode="sparse",
            es_base_url="http://localhost:9200",
        )


def test_elasticsearch_backend_accepts_sparse_with_elasticsearch_search_backend():
    s = Settings(
        persistence_backend="elasticsearch",
        search_backend="elasticsearch",
        retrieval_mode="sparse",
        es_base_url="http://localhost:9200",
    )
    assert s.search_backend == "elasticsearch"


def test_opensearch_search_backend_requires_url():
    with pytest.raises(ValueError, match="os_base_url is required"):
        Settings(search_backend="opensearch", retrieval_mode="dense")


def test_solr_search_backend_rejects_dense():
    with pytest.raises(ValueError, match="supports only retrieval_mode=sparse"):
        Settings(
            search_backend="solr",
            retrieval_mode="dense",
            solr_base_url="http://localhost:8983",
        )


def test_elasticsearch_backend_does_not_validate_sqlite_url():
    s = Settings(
        persistence_backend="elasticsearch",
        retrieval_mode="dense",
        es_base_url="http://localhost:9200",
        sqlite_url="postgres://ignored-in-es-mode",
    )
    assert s.sqlite_url == "postgres://ignored-in-es-mode"


def test_ollama_url_validator():
    with pytest.raises(ValueError):
        Settings(ollama_base_url="localhost:11434")


def test_chunk_overlap_lt_chars_validator():
    """Ensure overlap must be strictly less than chunk size."""
    with pytest.raises(ValueError):
        Settings(ingest_chunk_chars=100, ingest_chunk_overlap=100)


def test_ingest_batch_size_accepts_minimum():
    s = Settings(ingest_batch_size=1)
    assert s.ingest_batch_size == 1


def test_ingest_batch_size_accepts_maximum():
    s = Settings(ingest_batch_size=512)
    assert s.ingest_batch_size == 512


def test_ingest_batch_size_rejects_below_minimum():
    with pytest.raises(ValueError):
        Settings(ingest_batch_size=0)


def test_ingest_batch_size_rejects_above_maximum():
    with pytest.raises(ValueError):
        Settings(ingest_batch_size=513)


def test_log_level_is_normalized_to_uppercase():
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"


def test_debug_accepts_release_alias_from_string():
    s = Settings(debug="release")
    assert s.debug is False


def test_debug_accepts_development_alias_from_string():
    s = Settings(debug="development")
    assert s.debug is True


def test_data_dir_is_not_created_as_a_side_effect(tmp_path):
    d = tmp_path / "new-data-dir"
    assert not d.exists()
    s = Settings(data_dir=d)
    assert s.data_dir == d
    assert not d.exists()


def test_coordination_dir_prefers_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "coord-dir"
    s = Settings(data_dir=data_dir, sqlite_url=f"sqlite:///{tmp_path / 'app.db'}")
    assert s.get_coordination_dir() == data_dir.resolve()


def test_coordination_dir_uses_absolute_sqlite_parent_when_data_dir_is_relative(
    tmp_path, monkeypatch
):
    shared_db = tmp_path / "shared" / "app.db"
    shared_db.parent.mkdir(parents=True, exist_ok=True)
    wd_a = tmp_path / "worker-a"
    wd_b = tmp_path / "worker-b"
    wd_a.mkdir()
    wd_b.mkdir()

    monkeypatch.chdir(wd_a)
    s_a = Settings(data_dir=Path("coord"), sqlite_url=f"sqlite:///{shared_db}")
    dir_a = s_a.get_coordination_dir()

    monkeypatch.chdir(wd_b)
    s_b = Settings(data_dir=Path("coord"), sqlite_url=f"sqlite:///{shared_db}")
    dir_b = s_b.get_coordination_dir()

    assert dir_a == shared_db.parent.resolve()
    assert dir_b == shared_db.parent.resolve()
    assert dir_a == dir_b


def test_coordination_dir_uses_absolute_sqlite_parent_when_data_dir_is_default(tmp_path):
    s = Settings(data_dir=Path("data"), sqlite_url=f"sqlite:///{tmp_path / 'app.db'}")
    assert s.get_coordination_dir() == tmp_path.resolve()


def test_load_settings_from_yaml_resolves_relative_paths(tmp_path):
    config_dir = tmp_path / "config-root"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_dir: data",
                "index_path: data/index.faiss",
                "id_map_path: data/id_map.json",
                "sqlite_url: sqlite:///./data/app.db",
                "faq_csv: data/faq.csv",
                "eval_dataset_path: datasets/rag_eval_v1.jsonl",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings_from_yaml(config_file)

    assert settings.data_dir == (config_dir / "data").resolve()
    assert settings.index_path == str((config_dir / "data/index.faiss").resolve())
    assert settings.id_map_path == str((config_dir / "data/id_map.json").resolve())
    assert settings.sqlite_url == f"sqlite:///{(config_dir / 'data/app.db').resolve()}"
    assert settings.faq_csv == str((config_dir / "data/faq.csv").resolve())
    assert settings.eval_dataset_path == str((config_dir / "datasets/rag_eval_v1.jsonl").resolve())


def test_load_settings_from_yaml_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_settings_from_yaml(tmp_path / "missing.yaml")


def test_load_settings_from_yaml_rejects_non_mapping(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- not-a-mapping\n- still-not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top level"):
        load_settings_from_yaml(config_file)
