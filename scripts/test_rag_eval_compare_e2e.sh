#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

DATASET_PATH="$TMP_DIR/rag_eval_compare.jsonl"
PASS_JSON="$TMP_DIR/pass.json"
FAIL_JSON="$TMP_DIR/fail.json"
PASS_SPEC="$TMP_DIR/pass-spec.json"
FAIL_SPEC="$TMP_DIR/fail-spec.json"

cat >"$DATASET_PATH" <<'EOF'
{"type":"meta","dataset_id":"rag_eval_compare_smoke","schema_version":1}
{"type":"doc","external_id":"doc:alpha","source_id":"smoke","content":"alpha alpha alpha"}
{"type":"query","query":"alpha","relevant_external_ids":["doc:alpha"]}
EOF

cat >"$PASS_SPEC" <<'EOF'
{
  "k": 1,
  "baseline": {"retrieval_mode": "sparse"},
  "candidate": {"retrieval_mode": "sparse"},
  "thresholds": {
    "min_delta_ndcg": 0.0,
    "min_delta_map": 0.0,
    "min_delta_mrr": 0.0,
    "max_regression_precision": 0.0,
    "max_regression_recall": 0.0
  }
}
EOF

cat >"$FAIL_SPEC" <<'EOF'
{
  "k": 1,
  "baseline": {"retrieval_mode": "sparse"},
  "candidate": {"retrieval_mode": "sparse"},
  "thresholds": {
    "min_delta_ndcg": 0.1
  }
}
EOF

cd "$ROOT_DIR"

echo "[1/2] Expect PASS with identical sparse baseline/candidate and zero delta thresholds"
env DEBUG=false UV_CACHE_DIR="${UV_CACHE_DIR:-.uv_cache}" uv run rag-eval-compare \
  --dataset "$DATASET_PATH" \
  --spec "$PASS_SPEC" \
  --json-out "$PASS_JSON"

python - <<'PY' "$PASS_JSON"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["gate"]["passed"] is True
assert set(payload.keys()) == {"dataset_id", "k", "baseline", "candidate", "delta", "gate"}
print("PASS json validated")
PY

echo "[2/2] Expect FAIL when requiring an impossible positive delta from an identical candidate"
set +e
env DEBUG=false UV_CACHE_DIR="${UV_CACHE_DIR:-.uv_cache}" uv run rag-eval-compare \
  --dataset "$DATASET_PATH" \
  --spec "$FAIL_SPEC" \
  --json-out "$FAIL_JSON"
rc=$?
set -e

if [[ "$rc" -ne 1 ]]; then
  echo "Expected exit code 1 from placebo gate, got: $rc" >&2
  exit 1
fi

python - <<'PY' "$FAIL_JSON"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert payload["gate"]["passed"] is False
assert payload["gate"]["reasons"]
print("FAIL json validated")
PY

echo "rag-eval-compare e2e smoke passed"
