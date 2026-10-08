# Release Checklist

> Purpose: local release notes and pre-tag checklist for `rag-prototype`.
>
> Keep this file updated before cutting a public Git tag or GitHub Release.

## Pre-Tag Checklist

1. Freeze the release candidate:
   - worktree is clean
   - exact `target_ref` is recorded
   - branch policy is explicit (`master`, `develop`, or another release branch)
2. Validate version metadata:
   - `pyproject.toml` version matches the intended public version
   - tag naming follows the chosen convention (`vX.Y.Z` or `X.Y.Z`)
   - GitHub Release target points at the same commit that was tested
3. Validate runtime bootstrap:
   - `config.yaml` exists or is created from `config.example.yaml`
   - `rag-bootstrap` succeeds from a fresh checkout/copy
   - repeated `rag-bootstrap` is idempotent for sample data
4. Validate server smoke:
   - `/healthz` returns 200
   - `/` returns 200
   - `/openapi.json` returns 200
   - `/readyz` behavior is understood:
     - 200 when an LLM provider is configured and backend state is ready
     - 503 with actionable message when no LLM provider is configured
5. Run gates from the frozen ref:
   - `pytest -q`
   - `ruff check src tests`
   - `ruff format --check src tests`
   - `mypy src`
   - `lint-imports`
   - `pre-commit run --all-files`
   - `make smoke-embedding-api-wheel`

## Candidate: 4.0.0

The current package metadata identifies the next breaking candidate. This local
stabilization does not publish a tag, release, or package, and does not execute
data migrations. Existing `v3.0.0` tags and artifacts remain immutable.

Breaking changes relative to `v3.0.0`:

- The documented `performance` extra is removed. Select the declared extras;
  `monitoring` now installs Prometheus only. `dense-st` remains the optional
  SentenceTransformers extra.
- MCP requires a valid initialize request and strict tool arguments. Invalid
  arguments return JSON-RPC `-32602`; execution failures return tool content with
  `isError: true` instead of JSON-RPC `-32000`. Notifications have no response.
  Consumers must inspect the tool result as well as protocol errors.
- The typed embedding API retains its factory and service methods, but
  `model_key` now encodes the full embedding identity as a JSON string, and
  status includes `implementation_version`. Treat the key as opaque; do not
  parse the former colon-delimited representation. See
  [the installed API contract](embedding_integration.md).
- Library examples use `get_settings()` instead of importing a global
  `settings` instance. Imports no longer read YAML or initialize the database;
  explicit containers own their runtime resources and must be closed.
- Persistence and search now use local SQLite and FAISS/NumPy only. Remote
  Elasticsearch, OpenSearch, and Solr adapters and their settings were removed.
- Ingestion stores raw chunks and metadata separately. Document identity no
  longer includes the embedding model. Existing data needs a fresh empty data
  directory and full re-ingest; incremental ingest or vector-only rebuild can
  leave duplicate IDs and mixed content. This candidate does not delete data.
- The mutation journal stores active records in `active/` and terminal receipts
  in `done/`. Terminal receipts are retained for up to 30 days or 256 MiB, so
  exact `op_id` replay is bounded by that retention window. Old flat receipts
  remain read-only and replayable. Incomplete or corrupt records block writes
  and make `/readyz` return `503`; rebuilding an index does not repair them.

HTTP/MCP filter values must be nonempty arrays of nonblank strings. The transport
also recovers from JSON decoder `ValueError`, including integer literals beyond
Python's configured digit limit, so the next MCP request remains processable.
The canonical import contract accepts RepoGPT code-units v5 and
the generic rag-adapters payload; installed-artifact receipts establish the
specific tested producer combination.

For the retained code-units v5 boundary, canonical import accepts at most 5,000
nonblank documents per snapshot, 20,000 characters per document, 512 characters
per external ID, and 1,024 characters per source ID. RepoGPT can emit a valid v5
artifact beyond those consumer limits; RAG rejects it before importing. This
stabilization preserves the v5 contract and does not adopt the deferred v6
snapshot-policy migration. Validate the exact producer wheels before freezing
their candidate revisions.

Build each candidate's wheel and sdist with `uv build --out-dir <separate-dir>`;
pass one exact wheel from each build, retaining its SHA-256 and source revision:

```bash
make smoke-canonical-wheels \
  REPOGPT_WHEEL=/absolute/path/repogpt-0.10.0-py3-none-any.whl \
  ADAPTERS_WHEEL=/absolute/path/rag_adapters-0.2.0-py3-none-any.whl \
  RAG_WHEEL=/absolute/path/rag_prototype-4.0.0-py3-none-any.whl
```

This gate creates a fresh environment outside checkouts, installs only the
three selected wheels and their declared runtime dependencies, verifies import
origins and packaged schemas, and uses synthetic text with sparse local SQLite
retrieval. It exercises CLI import, idempotent replay, adapter failure evidence,
and refusal to replace a partial scope. It clears source injection and provider
credentials; it does not certify optional extractors, paid providers, or private
data. Dependency installation may require registry access; `UV_OFFLINE=1` uses
only cached dependencies. Freeze producer revisions together with
these exact artifact hashes before final acceptance.

The local `data/.mutation_journal-terminal-legacy-20261004T224154Z/` directory
retains 525 private legacy `COMMITTED` receipts with before-images, outside the
active journal. All 525 file hashes and sizes match the campaign custody
baseline captured on 2026-10-07. Its exact directory is excluded from Git; this
does not authorize deletion, upload, or replay into current data. These retained
files are not source acceptance evidence and do not certify a data migration.

## Historical boundary: 3.0.0

Breaking changes relative to 2.1.0:

- Move HTTP document ingestion from `POST /api/docs` to
  `POST /api/docs/ingest`.
- Move conversation import from `POST /api/docs/import` to
  `POST /api/docs/import-conversations`.
- Remove the `performance-cpu` extra; the `performance` extra still existed at
  the `v3.0.0` boundary and is removed by the candidate above.
- Require a boolean `debug` configuration value instead of historical string
  aliases.

At that boundary the typed embedding integration and MCP entrypoints retained
their preceding contracts. The 4.0.0 changes above supersede that compatibility
statement. No legacy aliases or new migration guarantee for historical SQLite
databases were introduced by 3.0.0.

The release also includes the already-implemented canonical import diagnostics,
HTTP/frontend hardening, and rejection of incomplete legacy remote enumeration
before destructive scope operations. Existing regression suites cover those
behaviors. Historical validation results below do not certify the current
candidate; record its exact commit, gates, and artifact hashes separately.

## Historical RC Notes: 2026-07-10

Planned release:

- Canonical promotion path: `develop` -> `main`.
- Package/tag version: `2.1.0` / `v2.1.0`.
- Historical `2.0.1` and `v1.3.0` tags remain untouched.
- `2.1.0` is intentionally forward-only because GitHub already published
  `2.0.1`, even though that tagged commit reports package version `1.3.0`.
- Release scope: strict `rag_ask` MCP input handling, explicit
  `RAG_CONFIG_PATH`, safe Makefile environment bootstrap, and the installed
  typed embedding integration API.

Validated locally before remote promotion:

- Full suite: 742 passed, 4 skipped, 87.41% branch coverage.
- Ruff, formatting, mypy, pre-commit/security hooks: passed.
- Import-linter: 5 contracts kept.
- Installed-wheel embedding API smoke: passed outside the checkout with no
  SentenceTransformers dependency.

Remote CI, Docker, Python 3.11, final `main` commit, and release artifact hashes
must be recorded after GitHub authentication is restored.

## RC Notes: 2026-04-27

Assumed RC:

- Branch: `develop`
- Commit: `908d6feaa53e7477bf88c7ddb1ab06c788782bbd`
- Package version: `1.3.0`
- Python support: `>=3.11,<3.13`

Reviewer verdict: `ship_with_notes`.

Validated local happy path:

- `uv venv .venv` used CPython 3.12.12.
- `uv sync --frozen --extra server` succeeded.
- `rag-bootstrap` succeeded.
- Resulting database contained 30 documents and 0 history entries.
- Re-running `rag-bootstrap` kept 30 documents.
- `rag-server` exposed `/healthz`, `/`, and `/openapi.json` successfully.
- `/readyz` returned 503 when no LLM provider was configured, with an actionable message.

Validated gates:

- `pytest -q`: 711 passed, 4 skipped, coverage 87.08%.
- `ruff check src tests`: passed.
- `ruff format --check src tests`: passed.
- `mypy src`: passed.
- `lint-imports`: 4 contracts kept.
- `pre-commit run --all-files`: passed in a temporary initialized copy.

Reviewer findings and handling:

- Addressed in this documentation pass:
  - Quickstart makes the `config.yaml` contract explicit.
  - Quickstart states that `/readyz` and `/api/ask` need an enabled LLM provider.
  - `make sync` / `make test` avoid `dense-st` by default; SentenceTransformers stays opt-in.
- Still open:
  - Release tags, GitHub Releases, package version, and default branch need reconciliation before the next public release.

Known scope limits:

- Docker Compose was not validated in this RC pass.
- External backends were not validated in this RC pass.
- The validated path was local `local_split` + sparse.
