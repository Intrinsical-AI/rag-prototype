from types import SimpleNamespace

from click.testing import CliRunner

from local_rag_backend.cli_commands.docs import docs_mutate
from local_rag_backend.core.errors import MutationRecoveryRequiredError


def test_mutation_recovery_error_is_actionable_without_traceback(tmp_path, monkeypatch):
    payload = tmp_path / "mutation.json"
    payload.write_text('{"upserts":[{"external_id":"one","content":"text"}]}')
    container = SimpleNamespace(build_docs_mutation_bundle=lambda **kw: SimpleNamespace(ports=None))
    monkeypatch.setattr(docs_mutate, "get_cli_container", lambda: container)
    monkeypatch.setattr(docs_mutate, "MutationCoordinator", lambda **kw: None)

    def fail(operation, **kwargs):
        raise MutationRecoveryRequiredError(
            "journal.jsonl op_id=op-1 state=pending requires recovery"
        )

    monkeypatch.setattr(docs_mutate, "run_cli_mutation", fail)
    result = CliRunner().invoke(docs_mutate.mutate_docs_cmd, ["--json", str(payload)])
    assert result.exit_code == 1
    assert "requires recovery" in result.output
    assert "journal.jsonl" in result.output
    assert "op-1" in result.output
    assert "Traceback" not in result.output
