# Notas internas de synergy

Este documento agrupa los packs y notas que dependen del workspace compartido `synergy`.
No forman parte del flujo de uso general del README ni de la guía de uso avanzado.

## Pack RepoGPT -> rag-prototype

Fixture compartida en `synergy`:

```bash
../synergy/synergy-up-search
bash ../synergy/scripts/repogpt_ingest_demo.sh
bash ../synergy/scripts/repogpt_eval_smoke.sh
bash ../synergy/scripts/repogpt_ingest_demo.sh --profile local_split
```

Notas:

* Shared fixture repo: `../synergy/fixtures/repogpt_eval_repo/`
* Cross-repo demos/smokes: `../synergy/scripts/repogpt_ingest_demo.sh`, `../synergy/scripts/repogpt_eval_smoke.sh`
* `../synergy` uses `elasticsearch` as the default workspace profile; this repo does not.
* Consumer-owned eval dataset: `datasets/repogpt_rag_eval_v1.jsonl`
* Maintained import/search coverage: `tests/e2e/test_repogpt_ingest_search_eval.py`
* Supported producer contract: `RepoGPT code-units` schema `4`
* Agent-facing status now exposes a structured runtime snapshot instead of raw config fields.
* Canonical import transport is shared across CLI, HTTP, and MCP; RepoGPT-specific checks live at the border.

## Vulnerability pilot pack

```bash
python ../synergy/scripts/vulns_batch_triage.py \
  --input ../synergy/vuln_pilot/prepared/pilot_small_v1.jsonl \
  --profile high_severity_python \
  --output /tmp/vuln-triage-high.jsonl

python ../synergy/scripts/vulns_ingest_rag.py \
  --input ../synergy/vuln_pilot/prepared/pilot_small_v1.jsonl \
  --payload-out /tmp/vuln-pilot.json \
  --no-import
```

Notas:

* Shared prepared snapshot: `../synergy/vuln_pilot/prepared/pilot_small_v1.jsonl`
* Cross-repo batch/import scripts: `../synergy/scripts/vulns_batch_triage.py`, `../synergy/scripts/vulns_ingest_rag.py`
* Consumer-owned eval dataset: `datasets/vuln_pilot_rag_eval_v1.jsonl`
* Maintained import/search coverage: `tests/e2e/test_vuln_pilot_ingest_search_eval.py`

## Dataset-specific notes

Dataset específico de RepoGPT:

* El dataset `datasets/repogpt_rag_eval_v1.jsonl` pertenece a `rag-prototype`, no a `RepoGPT`, porque define la barra de calidad del consumidor.

Dataset específico del piloto de vulnerabilidades:

* La fuente preparada compartida vive en `../synergy/vuln_pilot/prepared/pilot_small_v1.jsonl`.
* El dataset `datasets/vuln_pilot_rag_eval_v1.jsonl` pertenece a `rag-prototype`, no a `structured-research`, porque define la barra de calidad del consumidor.
* El perfil `cwe_78_focus` vive en `structured-research/config/vuln_triage/cwe_78_focus/`.

## Smoke e2e

```bash
bash scripts/test_rag_eval_compare_e2e.sh
```
