# Updated Hybrid Coding Plan: Codex-Led Local Direct Workers

## Summary

Use Codex/GPT-5.5 as the high-level planner, reviewer, implementation lead, and
final integrator. Local V100-hosted models are bounded helper workers invoked through
`scripts/hybrid-worker`: they receive explicit target files, return complete file
content through direct `/chat/completions`, and produce review-only artifacts from
an isolated workspace.

The production MVP default is `Qwen/Qwen2.5-Coder-7B-Instruct` via direct vLLM chat,
because direct generation passed the local MBPP pilot while Qwen Code file-edit mode
timed out on tool-use constraints. Qwen Code, OpenHands, and OpenCode remain optional
agentic research paths until separate benchmarks prove reliable tool use.

The active local-worker path is review-only. The local model never edits the real
repo; the coordinator writes generated content only inside a git worktree or copied
workspace, computes `diff.patch`, runs optional checks, and leaves any integration to
Codex review.

## Key Architecture

- Coordinator: Codex with `gpt-5.5` or `gpt-5.4`.
- Local direct worker CLI: `scripts/hybrid-worker`.
- Production MVP model: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Benchmark upgrade candidate: `Qwen/Qwen2.5-Coder-14B-Instruct`.
- Quality/research model: `Qwen/Qwen3-Coder-30B-A3B-Instruct`.
- Serving layer: vLLM OpenAI-compatible API.
- Baseline 7B GPU topology: `--partition=multigpu --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:v100-sxm2:1 --time=12:00:00`.
- Optional quality/research topology: 2 nodes with 4 V100s per node when available.
- Baseline 7B context target: `MAX_MODEL_LEN=16384`, `TENSOR_PARALLEL_SIZE=1`, `DTYPE=auto`.
- Control node: run Codex, Qwen Code clients, optional code-server, and one vLLM server on `127.0.0.1:8011`.
- Worker node: run second vLLM server on `<worker-node>:8012`.
- Baseline 7B starting script: `scripts/gpu_2node_qwen25coder7b_interactive.sh`.
- Optional 32B research starting script: `scripts/multigpu_2node_qwen25coder_interactive.sh`.
- Baseline 7B vLLM env: `/projects/aclab/liu.shu/envs/qwen25-vllm092-torch27-cu126/bin/python`.
- Optional 32B vLLM env default: `/projects/aclab/liu.shu/envs/qwen3-vllm-cu118/bin/python`.
- Cache root: `/projects/aclab/liu.shu/model-cache`.
- Outlines guided-decoding cache: the two-node launcher sets `OUTLINES_CACHE_DIR` to a node-local path, defaulting to `/tmp/$USER/outlines-cache-${SLURM_JOB_ID:-manual}-$(hostname -s)-$PORT`.
- Walltime guard: the 7B launcher starts `scripts/slurm_time_guard.sh` in the control shell to warn before Slurm terminates the allocation.

## Implementation Plan

- Keep Codex as the planner, reviewer, and final integrator.
- Use `scripts/hybrid-worker` for bounded direct-generation tasks with one explicit target file or at most three explicit target files.
- Use the 7B two-node script for the production MVP direct-generation baseline.
- Use 14B and Qwen3 30B-A3B only as benchmark or quality candidates until they beat the 7B local eval without unacceptable latency.
- Validate control-node access to both endpoints: `http://127.0.0.1:8011/v1` and `http://<worker-node>:8012/v1`.
- Confirm `/v1/models` reports `max_model_len: 16384` for both 7B endpoints after startup.
- Route through an endpoint registry or explicit `--endpoint`; health-check `/v1/models` before dispatch.
- Keep the runner mechanical: create isolated worktree/temp repo, copy dirty working trees when uncommitted tests or context exist, collect selected context, call direct chat with OpenAI `response_format: {"type": "json_object"}`, parse JSON, write generated files in the workspace, run tests, collect diff/logs, enforce allowed paths, and leave any human-visible conclusion to Codex review.
- Keep Qwen Code, OpenHands, OpenCode, parser/MCP, and tool-call experiments as optional research, not the first production path.

## Test Plan

- Run the 7B two-node Qwen2.5 launcher.
- Confirm `/v1/models` works for both control and worker endpoints from the control node.
- Confirm both 7B endpoints advertise `max_model_len: 16384`.
- Run one minimal `/v1/chat/completions` request against both endpoints.
- Run `scripts/check_hybrid_worker.py` for offline safety checks.
- Run `scripts/run_mbpp_single_file_eval.py --run-id mbpp-25` and aggregate results.
- Require at least 80% pass rate on the first 25 sanitized MBPP tasks.
- Require median generation latency below 10 seconds on the 7B baseline.
- Classify all failures as generation, parsing, test, timeout, endpoint, empty patch, or disallowed path.
- Run repo-fixture evals for single-file patches, reviewed test generation, and documentation generation.
- Treat Qwen Code single-worker smoke as a diagnostic gate only; do not promote it to the production path.

## Failed Multi-File Qwen Evidence

The artifact-only multi-file protocol exercise failed and is evidence against using
Qwen2.5-Coder 7B/vLLM as a practical implementation subagent:

- Proxy path: requests hit API 400 context-budget failures, timeout loops, and an
  incomplete or invalid one-file patch instead of a reliable multi-file result.
- No-proxy path: the model returned plain Markdown instead of executable OpenAI
  tool calls, so Qwen Code did not perform the requested edits.
- The only validated positive path remains narrow proxy-forced single-file smoke
  behavior under controlled artifact review.

## Codex Local Worker Dispatch Protocol

- Lead model owns architecture, task decomposition, integration, verification, and the final user-visible answer.
- Local workers handle bounded generation only when the task can be specified with one to three target files and clear allowed paths.
- Every local-worker task must include the repo, task text, target files, allowed paths, model profile, endpoint or endpoint registry, and optional test commands.
- Expected worker output is `summary.json`, generated files, `diff.patch`, raw model response, selected context, test logs, and a replay command.
- The local model does not receive tools and does not write to the real repo.
- The integration rule is that local workers return artifacts for Codex review; Codex owns any final merge, commit, or user-facing conclusion.

## Safe Integration Policy

The lead may integrate local-worker changes only after reviewing `diff.patch` and
confirming the worker stayed within the requested target files, changed paths pass
the allowed-path gate, tests/checks pass or failures are understood, and the diff is
small enough to review directly.

If ownership overlaps, tests fail without a clear diagnosis, files are dirty in conflicting ways, or a worker reports uncertainty, the lead treats the result as review-only and either resolves the issue locally or asks before proceeding. Worker output is never applied blindly.

## Protocol Test Cases

- Single-file implementation: `scripts/hybrid-worker` returns one replacement file and a patch; the real repo remains unchanged.
- Multi-file output outside `--allowed-path`: summary reports `patch_disallowed_paths`.
- Invalid model JSON: summary reports `invalid_model_output` and preserves raw output.
- Test failure: summary reports `tests_failed`, preserves artifacts, and leaves the real repo unchanged.
- Endpoint unavailable: summary reports `setup_error`, not an uncaught crash.
- Dirty repo with uncommitted reviewer tests: worker copies the working tree, runs those tests, and reports failures without touching the real repo.

## Operational Exercise

`scripts/check_hybrid_worker.py` exercises the deterministic safety cases without
requiring vLLM or GPUs. MBPP and repo-fixture evals exercise live local model quality.

## Assumptions

- 7B on two 1x V100 nodes is the default target for direct worker generation and endpoint diagnostics.
- 14B on 2 V100s per node is a benchmark candidate.
- Qwen3-Coder 30B-A3B is a quality/research candidate when 2x4 V100 capacity is available.
- The current Qwen Code runner milestone remains diagnostic single-worker reliability only.
- Qwen Code, OpenHands, OpenCode, and parser experiments are out of the active implementation workflow until a fresh compatibility proof shows stable delegated coding behavior.
- Login-node to compute-node networking is optional. The real target is control-node to worker-node communication inside one Slurm allocation.
