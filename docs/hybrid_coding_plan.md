# Updated Hybrid Coding Plan: Qwen2.5 Two-Node V100 Topologies

## Summary

Use Codex as the high-level planner/reviewer and Qwen2.5-Coder as the local implementation worker. The default development topology is a cheap 2-node V100 SXM2 allocation running `Qwen/Qwen2.5-Coder-7B-Instruct` with one GPU per node.

Keep the existing `Qwen/Qwen2.5-Coder-32B-Instruct` 2-node, 4-GPU-per-node path intact for scale-up and final validation after worker, proxy, runner, and artifact flow debugging works on 7B.

The Codex sub-agent hybrid dispatch protocol is documented and exercised:
when Codex should split work, how delegated tasks are labeled, what artifacts workers
return, and what conditions allow the lead to integrate results. The next use is a
broader protocol exercise across multi-file tasks or later Qwen/32B validation. It is
not a new Qwen multi-worker dispatcher script.

## Key Architecture

- Coordinator: Codex with `gpt-5.5` or `gpt-5.4`.
- Development worker model: `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Final validation worker model: `Qwen/Qwen2.5-Coder-32B-Instruct`.
- Serving layer: vLLM OpenAI-compatible API.
- Development GPU topology: `--partition=multigpu --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:v100-sxm2:1 --time=12:00:00`.
- Final validation GPU topology: `--partition=multigpu --nodes=2 --gres=gpu:v100-sxm2:4`.
- Development context target: `MAX_MODEL_LEN=16384`, `TENSOR_PARALLEL_SIZE=1`, `DTYPE=auto`.
- Final validation context target: `MAX_MODEL_LEN=32768`, `TENSOR_PARALLEL_SIZE=4`; lower to `16384` for a run if V100 KV-cache capacity or latency becomes a bottleneck.
- Control node: run Codex, Qwen Code clients, optional code-server, and one vLLM server on `127.0.0.1:8011`.
- Worker node: run second vLLM server on `<worker-node>:8012`.
- Development starting script: `scripts/gpu_2node_qwen25coder7b_interactive.sh`.
- Final validation starting script: `scripts/multigpu_2node_qwen25coder_interactive.sh`.
- Development vLLM env: `/projects/aclab/liu.shu/envs/qwen25-vllm092-torch27-cu126/bin/python`.
- Final validation vLLM env default: `/projects/aclab/liu.shu/envs/qwen3-vllm-cu118/bin/python`.
- Cache root: `/projects/aclab/liu.shu/model-cache`.
- Outlines guided-decoding cache: the two-node launcher sets `OUTLINES_CACHE_DIR` to a node-local path, defaulting to `/tmp/$USER/outlines-cache-${SLURM_JOB_ID:-manual}-$(hostname -s)-$PORT`.
- Walltime guard: the development launcher starts `scripts/slurm_time_guard.sh` in the control shell to warn before Slurm terminates the allocation.

## Implementation Plan

- Keep Codex as the orchestrator. Do not replace it with another agent framework.
- Use the 7B two-node script as the first operational launcher for day-to-day development.
- Use the 32B two-node script only after the 7B flow passes endpoint, proxy, and runner smoke tests.
- Validate control-node access to both endpoints: `http://127.0.0.1:8011/v1` and `http://<worker-node>:8012/v1`.
- Confirm `/v1/models` reports `max_model_len: 16384` for both 7B endpoints after startup.
- Confirm `/v1/models` reports `max_model_len: 32768` for both 32B endpoints during final validation.
- Configure Qwen Code to use the available vLLM endpoint through OpenAI-compatible settings or CLI flags.
- Keep the thin worker runner mechanical: create isolated worktree/temp repo, call Qwen Code headless, run tests, collect diff/logs, enforce allowed paths, and leave final integration to Codex review.
- Treat Qwen multi-worker CLI dispatch as a later option. The next protocol use is broader multi-file execution: lead-owned decomposition, lane-labeled delegation, explicit ownership, worker result packets, and lead-reviewed integration; or later Qwen/32B validation.
- Defer Qwen3-Coder and MCP until Qwen2.5 V100 workflow is stable.

## Test Plan

- Run the 7B two-node Qwen2.5 launcher.
- Confirm `/v1/models` works for both control and worker endpoints from the control node.
- Confirm both 7B endpoints advertise `max_model_len: 16384`.
- Run one minimal `/v1/chat/completions` request against both endpoints.
- Start the tool-choice proxy against the worker endpoint and run the single-worker smoke gate.
- Run one isolated coding task end-to-end: Codex writes task, Qwen implements in temp worktree, tests run, Codex reviews/applies diff.
- Repeat endpoint and smoke validation on the 32B topology before treating quality-sensitive results as final.

## Codex Hybrid Dispatch Protocol

- Lead model owns architecture, task decomposition, integration, verification, and the final user-visible answer.
- Sub-agents handle bounded implementation slices, focused tests, docs capture, repetitive edits, or independent repo exploration when the user explicitly selects `Hybrid mode` or otherwise authorizes delegation.
- Every delegated task must be labeled `blocking`, `integration`, or `background`.
- Every dispatch packet must include the task lane, exact ownership of files/modules/responsibility, expected output, concurrency warning, and integration rule.
- Expected worker output is changed files, concise summary, tests or checks run, and unresolved issues with logs or reproduction details.
- The concurrency warning must say the worker is not alone in the codebase, must not revert edits made by others, and must adapt to concurrent changes.
- The integration rule is that workers return results for lead review; the lead owns any final merge, commit, or user-facing conclusion.

## Safe Integration Policy

The lead may integrate worker changes without asking again only when the user explicitly selected `Hybrid mode`, the worker stayed within ownership, changes do not overlap with other edits, tests/checks pass or failures are understood, and the diff is small enough to review directly.

If ownership overlaps, tests fail without a clear diagnosis, files are dirty in conflicting ways, or a worker reports uncertainty, the lead treats the result as review-only and either resolves the issue locally or asks before proceeding. Worker output is never applied blindly.

## Protocol Test Cases

- Small task without `Hybrid mode`: lead works locally and spawns no sub-agents.
- Medium task with `Hybrid mode`: lead spawns at least one bounded sub-agent with lane and ownership, continues non-overlapping work, then reviews returned changes before finalizing.
- Docs or test side work: lead delegates as `background`; routine success is collected at a natural boundary, while unresolved failures are reported with diagnosis and next step.
- Conflicting worker output: lead does not auto-integrate; final report names the conflict, affected files, and recommended next step.

## Operational Exercise

The protocol has been exercised under session-level `Hybrid mode`: a small change stayed local-only, a medium task used a bounded `integration` worker with explicit ownership, and review side work was delegated as `background`. Lead review remains required before any worker output is integrated.

## Assumptions

- 7B on two 1x V100 nodes is the default target for runner, proxy, endpoint, and artifact-flow development.
- 32B on two 4x V100 nodes is the final validation target for quality-sensitive work.
- The current Qwen runner milestone remains single-worker reliability; Codex sub-agent dispatch is documented, exercised, and ready. Next use is a broader multi-file protocol exercise or later Qwen/32B validation.
- Qwen2.5-Coder-32B on 4x V100 should use the 32k context target first; if startup fails with KV-cache capacity errors, retry the launcher with `MAX_MODEL_LEN=16384`.
- Qwen3-Coder is out of scope for v1 due to likely GPU/resource constraints.
- Login-node to compute-node networking is optional. The real target is control-node to worker-node communication inside one Slurm allocation.
