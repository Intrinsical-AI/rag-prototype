from __future__ import annotations

from types import SimpleNamespace

from local_rag_backend.core.use_cases import docs_ingest


def test_ingest_identity_does_not_depend_on_embedding_mode(monkeypatch):
    captured = []

    class FakeRepo:
        def get_tombstoned_external_ids(self, _external_ids):
            return set()

    class FakeCoordinator:
        def __init__(self, *, settings_obj, ports):
            _ = settings_obj, ports

        def execute(self, intent):
            captured.append(intent)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(external_id=it.external_id, id="doc-1") for it in intent.upserts
                ]
            )

    monkeypatch.setattr(docs_ingest, "MutationCoordinator", FakeCoordinator)
    settings = SimpleNamespace(
        retrieval_mode="sparse",
        openai_api_key=None,
        openai_embedding_model="embedding-v1",
        st_embedding_model="embedding-local",
        ingest_chunker_version="chars_v1",
        ingest_chunk_chars=1000,
        ingest_chunk_overlap=0,
    )
    ports = SimpleNamespace(doc_repo_factory=FakeRepo)
    raw = "  Case Sensitive\n  "

    assert docs_ingest.ingest_docs_sync(texts=[raw], settings_obj=settings, ports=ports) == [
        "doc-1"
    ]
    settings.retrieval_mode = "dense"
    settings.openai_api_key = "test"
    assert docs_ingest.ingest_docs_sync(texts=[raw], settings_obj=settings, ports=ports) == [
        "doc-1"
    ]

    first, second = captured
    assert first.upserts[0].content == raw
    assert first.upserts[0].external_id == second.upserts[0].external_id
    assert first.upserts[0].source_id == second.upserts[0].source_id
    assert "emb=" not in str(first.upserts[0].source_id)
