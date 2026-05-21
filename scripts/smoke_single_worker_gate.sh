#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:18012/v1}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/tmp/hybrid_agent_single_worker_smoke}
SMOKE_PATH=${SMOKE_PATH:-docs/worker_runner_smoke.md}
SMOKE_TEXT=${SMOKE_TEXT:-worker runner smoke ok}
TIMEOUT=${TIMEOUT:-300}
MODEL_TIMEOUT=${MODEL_TIMEOUT:-20}

cd "$REPO_ROOT"

PROXY_MODELS_URL="${ENDPOINT%/}/models"
if ! curl --noproxy "*" -fsS --max-time "$MODEL_TIMEOUT" "$PROXY_MODELS_URL" >/dev/null; then
  echo "Proxy endpoint is not ready: $PROXY_MODELS_URL" >&2
  exit 1
fi

RUNNER_STDOUT=$(mktemp "${TMPDIR:-/tmp}/single-worker-smoke.stdout.XXXXXX")
RUNNER_STDERR=$(mktemp "${TMPDIR:-/tmp}/single-worker-smoke.stderr.XXXXXX")
if ! scripts/run_worker_task.py \
  --repo "$REPO_ROOT" \
  --endpoint "$ENDPOINT" \
  --approval-mode yolo \
  --task "Create ${SMOKE_PATH} containing exactly one line: ${SMOKE_TEXT}" \
  --allowed-path "$SMOKE_PATH" \
  --test-command "test \"\$(cat ${SMOKE_PATH})\" = \"${SMOKE_TEXT}\"" \
  --timeout "$TIMEOUT" \
  --artifact-root "$ARTIFACT_ROOT" \
  >"$RUNNER_STDOUT" 2>"$RUNNER_STDERR"; then
  cat "$RUNNER_STDOUT"
  cat "$RUNNER_STDERR" >&2
  exit 1
fi

SUMMARY_JSON=$(cat "$RUNNER_STDOUT")

printf '%s\n' "$SUMMARY_JSON"

SUMMARY_FILE=$(
  SUMMARY_JSON="$SUMMARY_JSON" python -c 'import json, os; print(json.loads(os.environ["SUMMARY_JSON"])["artifact_dir"] + "/summary.json")'
)

SUMMARY_FILE="$SUMMARY_FILE" python - <<'PY'
import json
import os
import sys

summary_file = os.environ["SUMMARY_FILE"]
with open(summary_file, "r") as handle:
    summary = json.load(handle)

errors = []
if summary.get("ok") is not True:
    errors.append("summary ok is not true")
if summary.get("qwen_returncode") != 0:
    errors.append("qwen_returncode is not 0")
qwen = summary.get("qwen") or {}
if qwen.get("timeout_occurred"):
    errors.append("qwen timed out")
if summary.get("disallowed_paths"):
    errors.append("summary contains disallowed paths")
if not summary.get("changed_paths"):
    errors.append("summary changed_paths is empty")
if not summary.get("patch_sha256"):
    errors.append("summary patch_sha256 is missing")
for test in summary.get("tests") or []:
    if test.get("returncode") != 0:
        errors.append("test failed: {}".format(test.get("command")))

if errors:
    print("single-worker smoke gate failed:", file=sys.stderr)
    for error in errors:
        print("- " + error, file=sys.stderr)
    print("summary: " + summary_file, file=sys.stderr)
    sys.exit(1)

print("single-worker smoke gate passed: {}".format(summary_file))
PY
