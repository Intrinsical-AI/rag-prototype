from __future__ import annotations

from local_rag_backend.composition.runtime import build_runtime_snapshot
from local_rag_backend.settings import Settings


def test_runtime_snapshot_exposes_narrow_agent_surface(tmp_path) -> None:
    data_dir = tmp_path / "rag-data"
    index_path = tmp_path / "rag-index.faiss"
    id_map_path = tmp_path / "rag-id-map.json"
    eval_dataset_path = tmp_path / "rag-eval.jsonl"
    settings_obj = Settings(
        app_host="127.0.0.1",
        app_port=9090,
        debug=True,
        enable_monitoring=True,
        api_key="secret",
        public_bind_requires_api_key=False,
        persistence_backend="elasticsearch",
        es_base_url="https://es.local",
        search_backend="elasticsearch",
        retrieval_mode="hybrid",
        vector_backend="faiss",
        mutation_recovery_enabled=False,
        data_dir=data_dir,
        index_path=str(index_path),
        id_map_path=str(id_map_path),
        eval_dataset_path=str(eval_dataset_path),
        ollama_enabled=True,
        openai_api_key="openai-key",
        openrouter_enabled=True,
        openrouter_api_key="openrouter-key",
    )

    snapshot = build_runtime_snapshot(settings_obj)
    payload = snapshot.to_dict()

    assert payload["topology"] == {
        "host": "127.0.0.1",
        "port": 9090,
        "debug": True,
        "enable_monitoring": True,
    }
    assert payload["safety"] == {
        "public_bind_requires_api_key": False,
        "api_key_configured": True,
        "mutation_recovery_enabled": False,
    }
    assert payload["backends"] == {
        "persistence": "elasticsearch",
        "search": "elasticsearch",
        "retrieval": "hybrid",
        "vector": "faiss",
        "storage_profile": "",
    }
    assert payload["paths"] == {
        "data_dir": str(data_dir),
        "index_path": str(index_path),
        "id_map_path": str(id_map_path),
        "eval_dataset_path": str(eval_dataset_path),
    }
    assert payload["llm"] == {
        "ollama_enabled": True,
        "openai_enabled": True,
        "openrouter_enabled": True,
    }
