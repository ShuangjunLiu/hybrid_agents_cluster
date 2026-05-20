# Hybrid Coding V1 Usage

This repo now has the V1 control-plane tools for the Qwen2.5 two-node V100 setup.

## Progress Tracking

Track current research status, evidence, and next actions in `/home/liu.shu/knowledge-vault/02-research/2026-05-20-qwen25-vllm-qwen-code-tools/index.md`.

## 1. Discover Slurm Endpoints

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

## 2. Validate vLLM

Validate both endpoints. The worker URL is auto-discovered from `SLURM_JOB_NODELIST` when available.

```bash
scripts/validate_vllm_endpoints.py --timeout 60
```

For strict context validation:

```bash
scripts/validate_vllm_endpoints.py --timeout 60 --expected-max-model-len 32768
```

For automation:

```bash
scripts/validate_vllm_endpoints.py --timeout 60 --json
```

## 3. Run One Isolated Worker Task

The worker runner never edits the target repo directly. It creates an isolated workspace, runs Qwen Code there, then writes artifacts under `/tmp/hybrid_agent_tasks` by default.

```bash
scripts/run_worker_task.py \
  --repo /path/to/repo \
  --endpoint http://127.0.0.1:8011/v1 \
  --task "Implement the requested small change." \
  --allowed-path "src/**" \
  --test-command "pytest"
```

Artifacts:

- `task.md`: the submitted task.
- `run.log`: Qwen Code stdout/stderr.
- `test.log`: test command output, when tests are provided.
- `diff.patch`: generated patch.
- `summary.json`: machine-readable status.
- `workspace/`: isolated workspace containing generated changes.

Use the worker endpoint by changing `--endpoint`:

```bash
scripts/run_worker_task.py \
  --repo /path/to/repo \
  --endpoint http://d1011:8012/v1 \
  --task "Implement the requested small change."
```

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
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --prompt "Reply exactly ok" \
  --output-format text
```

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

The proxy only rewrites `/v1/chat/completions` JSON requests that contain `tools` and do not already set `tool_choice`.

By default the proxy stops forcing tool calls after a successful mutating tool result from `edit` or `write_file`. This matters for Qwen Code: forcing every turn can make the model keep calling tools after the edit is done, while stopping after the edit lets Qwen Code produce its final text answer and exit. To disable this guard for debugging:

```bash
scripts/openai_tool_choice_proxy.py \
  --listen-port 18011 \
  --upstream http://127.0.0.1:8011 \
  --stop-after-tool ""
```

The proxy also patches streaming tool-call responses from vLLM when they contain tool-call deltas but no final non-null `finish_reason`, inserting `finish_reason: "tool_calls"` before the usage or `[DONE]` event. Without that patch, Qwen Code can fail with `Model stream ended without a finish reason`.
