# Hybrid Coding V1 Usage

This repo now has the V1 control-plane tools for the Qwen2.5 two-node V100 setup.
Use the 7B, 2-node, 1x V100-per-node topology for day-to-day development and smoke
debugging. Keep the 32B, 2-node, 4x V100-per-node topology for scale-up and final
validation.

## Progress Tracking

Track current research status, evidence, and next actions in `/home/liu.shu/knowledge-vault/02-research/2026-05-20-qwen25-vllm-qwen-code-tools/index.md`.

## 1. Launch The 7B Development Topology

From any login-node directory:

```bash
source /home/liu.shu/env_sh/gpu_2node_qwen25coder7b_interactive.sh
```

Defaults:

- Slurm request: `--partition=multigpu --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:v100-sxm2:1 --time=12:00:00 --cpus-per-task=8 --mem=32G`.
- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Context and serving knobs: `MAX_MODEL_LEN=16384 TENSOR_PARALLEL_SIZE=1 DTYPE=auto`.
- vLLM env: `/projects/aclab/liu.shu/envs/qwen25-vllm092-torch27-cu126/bin/python`.
- Ports: control `8011`, worker `8012`.
- Repo root used by helper scripts: `/projects/aclab/liu.shu/codePool/hybrid_agents`, override with `HYBRID_AGENTS_REPO_ROOT` only if this repo moves.
- Walltime reminders: `scripts/slurm_time_guard.sh` starts in the control shell and warns at 60, 30, 15, 5, and 1 minutes remaining.

For a shorter public-GPU fallback, override the partition and walltime:

```bash
PARTITION=gpu-short SLURM_TIME=01:00:00 source /home/liu.shu/env_sh/gpu_2node_qwen25coder7b_interactive.sh
```

The default uses `multigpu` for longer coding sessions while still requesting only one
V100 per node:

```bash
PARTITION=multigpu source /home/liu.shu/env_sh/gpu_2node_qwen25coder7b_interactive.sh
```

Customize walltime reminders with environment variables before launching:

```bash
SLURM_TIME_GUARD_WARN_MINUTES=120,60,30,10,5,1 \
SLURM_TIME_GUARD_HOOK='git -C /projects/aclab/liu.shu/codePool/hybrid_agents status --short' \
source /home/liu.shu/env_sh/gpu_2node_qwen25coder7b_interactive.sh
```

The hook is optional. The default behavior is just terminal reminders plus a log under
`/tmp/$USER/slurm-time-guard-$SLURM_JOB_ID.log`.
The control-node shell starts in the repo root, and the launcher prints absolute smoke
commands so they work even when the allocation was started from `$HOME`.

As a last-resort endpoint-specific debugging fallback, run two independent 1-node jobs
and test each endpoint separately. Defer cross-node orchestration checks until a real
2-node allocation is available.

## 2. Discover Slurm Endpoints

Run inside the active two-node Slurm allocation:

```bash
scripts/discover_slurm_nodes.sh
```

Expected output:

```text
CONTROL_NODE=d1010
WORKER_NODE=d1011
CONTROL_ENDPOINT=http://127.0.0.1:8011/v1
WORKER_ENDPOINT=http://d1011:8012/v1
```

## 3. Validate vLLM

Validate both endpoints. The worker URL is auto-discovered from `SLURM_JOB_NODELIST` when available.

```bash
$HYBRID_AGENTS_REPO_ROOT/scripts/validate_vllm_endpoints.py --timeout 60
```

For strict context validation:

```bash
$HYBRID_AGENTS_REPO_ROOT/scripts/validate_vllm_endpoints.py --timeout 60 --expected-max-model-len 16384
```

For automation:

```bash
$HYBRID_AGENTS_REPO_ROOT/scripts/validate_vllm_endpoints.py --timeout 60 --json
```

For the full single-worker reliability gate, run the proxy stop-condition checks,
validate vLLM, then run the live worker smoke:

```bash
$HYBRID_AGENTS_REPO_ROOT/scripts/check_proxy_stop_conditions.py
$HYBRID_AGENTS_REPO_ROOT/scripts/check_worker_runner_review_apply.py
$HYBRID_AGENTS_REPO_ROOT/scripts/validate_vllm_endpoints.py --timeout 60 --expected-max-model-len 16384 --json
WORKER_NODE="${WORKER_NODE:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '2p')}"
WORKER_NODE="$WORKER_NODE" $HYBRID_AGENTS_REPO_ROOT/scripts/start_worker_tool_proxy.sh
ENDPOINT=http://127.0.0.1:18012/v1 $HYBRID_AGENTS_REPO_ROOT/scripts/smoke_single_worker_gate.sh
```

The smoke gate requires `summary.json` to report `ok: true`, `qwen_returncode: 0`,
no timeout, no disallowed paths, a non-empty patch, and passing tests. A patch created
before a Qwen timeout is still a failed smoke.

Acceptance: both `/v1/models` endpoints work, simple chat returns `ok`, worker smoke
reports `ok: true`, and no request times out.

## 4. Run One Isolated Worker Task

The worker runner never edits the target repo directly. It creates an isolated workspace, runs Qwen Code there, then writes artifacts under `/tmp/hybrid_agent_tasks` by default. Each worker run or patch review also appends one JSONL registry record to `/tmp/hybrid_agent_tasks/runs.jsonl` unless `--run-registry` overrides it.

```bash
scripts/run_worker_task.py \
  --repo /path/to/repo \
  --endpoint http://127.0.0.1:8011/v1 \
  --task "Implement the requested small change." \
  --approval-mode yolo \
  --allowed-path "src/**" \
  --test-command "pytest"
```

Artifacts:

- `task.md`: the submitted task.
- `run_metadata.json`: endpoint/model/run metadata plus a replay command.
- `run.log`: Qwen Code stdout/stderr.
- `test.log`: test command output, when tests are provided.
- `diff.patch`: generated patch.
- `summary.json`: stable machine-readable status, including `schema_version`, `failure_class`, `failure_reasons`, command result summaries, changed paths, disallowed paths, test results, and patch hash.
- `workspace/`: isolated workspace containing generated changes.

Inspect recent worker runs from the registry:

```bash
scripts/inspect_worker_runs.py --artifact-root /tmp/hybrid_agent_tasks --limit 10
```

Show the review details for a specific artifact directory:

```bash
scripts/inspect_worker_runs.py --show /tmp/hybrid_agent_tasks/20260521T120000Z
```

For automation, add `--json` to either command.

Timeout cleanup note: worker subprocesses must run in their own process group.
If only the parent Qwen Code process is killed on timeout, child processes can
keep inherited stdout/stderr pipes open, causing `proc.communicate()` to keep
waiting even though the parent is gone. The runner therefore starts subprocesses
in a new session and terminates the process group before collecting stdout/stderr.

Use the worker endpoint by changing `--endpoint`:

```bash
scripts/run_worker_task.py \
  --repo /path/to/repo \
  --endpoint http://d1011:8012/v1 \
  --task "Implement the requested small change."
```

The runner defaults to `--approval-mode yolo` because headless Qwen Code may refuse some
tools in `auto-edit` mode with a warning that automatic tool execution requires YOLO mode.
The workspace is isolated and the patch gate still controls what can be applied to the
target repo.

The runner also caps Qwen Code output with `QWEN_CODE_MAX_OUTPUT_TOKENS=1024` by default.
Set `QWEN_CODE_MAX_OUTPUT_TOKENS` in the environment or pass
`--qwen-max-output-tokens` for larger tasks. This is runner-local and does not modify
global Qwen Code configuration.

Context budget failure note: a previous 7B worker smoke failed because Qwen Code's
tool-use prompt plus requested completion budget exceeded the served vLLM context
window. vLLM effectively requires `prompt_tokens + requested_max_tokens <=
max_model_len`; with the old `MAX_MODEL_LEN=8192`, Qwen Code's large tool prompt and
output budget produced HTTP 400 before generation. The current mitigation is serving
the 7B development topology with `MAX_MODEL_LEN=16384` and keeping the runner-local
Qwen output budget capped at `1024` by default. For larger tasks, first confirm the
served `max_model_len` and available token budget before raising the cap.

## 5. Review Or Apply A Patch

By default the worker runner only returns artifacts. To review a patch against an
allowed-path gate without running Qwen Code:

```bash
scripts/run_worker_task.py \
  --repo /path/to/repo \
  --review-patch /tmp/hybrid_agent_tasks/20260520T120000Z/diff.patch \
  --allowed-path "src/**"
```

To apply a generated or reviewed patch automatically, pass `--apply`. The runner refuses
to apply when Qwen Code failed or timed out, a test command failed, a changed path is not
allowed, the patch is empty, the target repo has uncommitted tracked changes, or
`git apply --check` fails. Use `--allow-dirty-apply` only when the dirty tracked state is
intentional and already understood.

## Current Compatibility Note

On Explorer, Qwen Code may route even local vLLM traffic through the cluster proxy unless proxy variables are removed. If Qwen Code fails with:

```text
[API Error: Connection error. (cause: fetch failed)]
```

run it with proxy variables unset and `NO_PROXY='*' no_proxy='*'`.

Useful debug command:

```bash
env -u http_proxy -u https_proxy -u ftp_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u FTP_PROXY \
  -u ALL_PROXY -u all_proxy \
  -u npm_config_proxy -u npm_config_http_proxy -u npm_config_https_proxy \
  -u NPM_CONFIG_PROXY -u NPM_CONFIG_HTTP_PROXY -u NPM_CONFIG_HTTPS_PROXY \
  NO_PROXY='*' no_proxy='*' \
  qwen --bare --debug \
  --openai-logging \
  --openai-logging-dir /tmp/qwen-openai-debug \
  --auth-type openai \
  --openai-base-url http://127.0.0.1:8011/v1 \
  --openai-api-key EMPTY \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --prompt "Reply exactly ok" \
  --output-format text
```

For 32B final validation, use `--model Qwen/Qwen2.5-Coder-32B-Instruct` and validate
with `--expected-max-model-len 32768`.

## 32B Scale-Up And Final Validation

After the 7B development path passes endpoint, proxy, and smoke-gate checks, launch the
existing 32B topology:

```bash
source scripts/multigpu_2node_qwen25coder_interactive.sh
```

The 32B launcher keeps the existing defaults: `Qwen/Qwen2.5-Coder-32B-Instruct`,
`MAX_MODEL_LEN=32768`, `TENSOR_PARALLEL_SIZE=4`, and 4 V100 SXM2 GPUs per node on
`multigpu`.

## Tool-Choice Proxy Experiment

For Qwen2.5-Coder through vLLM, Qwen Code may send OpenAI `tools` while the model returns tool-looking text instead of parsed OpenAI `tool_calls`. Use the local proxy to inject `tool_choice: "required"` for chat requests that include tools:

```bash
scripts/openai_tool_choice_proxy.py \
  --listen-port 18011 \
  --upstream http://127.0.0.1:8011
```

Then point Qwen Code or the worker runner at:

```text
http://127.0.0.1:18011/v1
```

For the standard worker path, prefer the startup helper. It starts the proxy, waits for
`/v1/models`, writes a pid and log under `/tmp/$USER`, and prints the endpoint value:

```bash
WORKER_NODE=d1011 scripts/start_worker_tool_proxy.sh
```

The proxy only rewrites `/v1/chat/completions` JSON requests that contain `tools` and do not already set `tool_choice`.

Operational rule: inject `tool_choice: "required"` only until a successful mutating tool
result is observed. This stop-after-tool guard is part of the design, not just a timeout
workaround: Qwen2.5-Coder/vLLM needs forced tool choice to emit parsed OpenAI
`tool_calls`, while Qwen Code needs forced tool choice to stop after the edit so it can
produce its final text answer and exit. Without this boundary, a completed edit can turn
into a post-edit loop and eventually `qwen_timeout`.

By default the proxy stops forcing tool calls after a successful mutating tool result from
`edit`, `write_file`, or a mutating `run_shell_command`. Successful `edit` outcomes include
messages such as `Created new file`, `created`, `updated`, `modified`, and `replaced`, so
new-file edits stop the loop the same way replacements do. To disable this guard for
debugging:

```bash
scripts/openai_tool_choice_proxy.py \
  --listen-port 18011 \
  --upstream http://127.0.0.1:8011 \
  --stop-after-tool ""
```

The proxy also patches streaming tool-call responses from vLLM when they contain tool-call deltas but no final non-null `finish_reason`, inserting `finish_reason: "tool_calls"` before the usage or `[DONE]` event. Without that patch, Qwen Code can fail with `Model stream ended without a finish reason`.

If the smoke gate fails with an unclear Qwen Code, vLLM, proxy, or tool-calling error,
search current upstream docs and issue reports with the exact warning/error text and
the local context before continuing local experiments.

## Outlines Cache for Guided Decoding

vLLM's tool-call guided decoding path uses Outlines, and in the inspected version `XDG_CACHE_HOME`
was not enough to relocate its disk cache (it still defaulted to `~/.cache/outlines`).
The two-node launchers therefore set `OUTLINES_CACHE_DIR` explicitly for each node.
The worker-node failure that reported `sqlite3.OperationalError: locking protocol` was
specific to this Outlines/diskcache SQLite cache path, not basic worker connectivity.

The vLLM launchers set a safe node-local default:

```bash
OUTLINES_CACHE_DIR=/tmp/$USER/outlines-cache-${SLURM_JOB_ID:-manual}-$(hostname -s)-$PORT
```

Callers normally do not need to set it. Override `OUTLINES_CACHE_DIR` only when you need
custom cache placement, and keep it on node-local storage for tool-call guided decoding.
