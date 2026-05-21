# Updated Hybrid Coding Plan: Qwen2.5 Two-Node V100 Topologies

## Summary

Use Codex as the high-level planner/reviewer and Qwen2.5-Coder as the local implementation worker. The default development topology is a cheap 2-node V100 SXM2 allocation running `Qwen/Qwen2.5-Coder-7B-Instruct` with one GPU per node.

Keep the existing `Qwen/Qwen2.5-Coder-32B-Instruct` 2-node, 4-GPU-per-node path intact for scale-up and final validation after worker, proxy, runner, and artifact flow debugging works on 7B.

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
- Add only a thin worker runner after manual flow works. Its job is mechanical: create isolated worktree/temp repo, call Qwen Code headless, run tests, collect diff/logs, enforce allowed paths.
- Defer Qwen3-Coder and MCP until Qwen2.5 V100 workflow is stable.

## Test Plan

- Run the 7B two-node Qwen2.5 launcher.
- Confirm `/v1/models` works for both control and worker endpoints from the control node.
- Confirm both 7B endpoints advertise `max_model_len: 16384`.
- Run one minimal `/v1/chat/completions` request against both endpoints.
- Start the tool-choice proxy against the worker endpoint and run the single-worker smoke gate.
- Run one isolated coding task end-to-end: Codex writes task, Qwen implements in temp worktree, tests run, Codex reviews/applies diff.
- Repeat endpoint and smoke validation on the 32B topology before treating quality-sensitive results as final.

## Assumptions

- 7B on two 1x V100 nodes is the default target for runner, proxy, endpoint, and artifact-flow development.
- 32B on two 4x V100 nodes is the final validation target for quality-sensitive work.
- Qwen2.5-Coder-32B on 4x V100 should use the 32k context target first; if startup fails with KV-cache capacity errors, retry the launcher with `MAX_MODEL_LEN=16384`.
- Qwen3-Coder is out of scope for v1 due to likely GPU/resource constraints.
- Login-node to compute-node networking is optional. The real target is control-node to worker-node communication inside one Slurm allocation.
