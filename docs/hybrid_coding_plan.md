# Updated Hybrid Coding Plan: Qwen2.5 Two-Node V100 Topology

## Summary

Use Codex as the high-level planner/reviewer and Qwen2.5-Coder as the local implementation worker. The primary runtime topology is one 2-node `multigpu` Slurm allocation using V100 SXM2 nodes.

This avoids depending on newer GPUs for Qwen3-Coder and keeps the system aligned with resources that are more likely to be available.

## Key Architecture

- Coordinator: Codex with `gpt-5.5` or `gpt-5.4`.
- Worker model: `Qwen/Qwen2.5-Coder-32B-Instruct`.
- Serving layer: vLLM OpenAI-compatible API.
- GPU topology: `--partition=multigpu --nodes=2 --gres=gpu:v100-sxm2:4`.
- Default context target: `MAX_MODEL_LEN=32768` per vLLM launcher default; lower to `16384` for a run if V100 KV-cache capacity or latency becomes a bottleneck.
- Control node: run Codex, Qwen Code clients, optional code-server, and one vLLM server on `127.0.0.1:8011`.
- Worker node: run second vLLM server on `<worker-node>:8012`.
- Starting script: `/home/liu.shu/env_sh/multigpu_2node_qwen25coder_interactive.sh`.
- vLLM env: `/projects/aclab/liu.shu/envs/qwen3-vllm-cu118/bin/python`.
- Cache root: `/projects/aclab/liu.shu/model-cache`.

## Implementation Plan

- Keep Codex as the orchestrator. Do not replace it with another agent framework.
- Use the existing two-node script as the first operational launcher.
- Validate control-node access to both endpoints: `http://127.0.0.1:8011/v1` and `http://<worker-node>:8012/v1`.
- Confirm `/v1/models` reports `max_model_len: 32768` for both endpoints after startup.
- Configure Qwen Code to use the available vLLM endpoint through OpenAI-compatible settings or CLI flags.
- Add only a thin worker runner after manual flow works. Its job is mechanical: create isolated worktree/temp repo, call Qwen Code headless, run tests, collect diff/logs, enforce allowed paths.
- Defer Qwen3-Coder and MCP until Qwen2.5 V100 workflow is stable.

## Test Plan

- Run the existing two-node Qwen2.5 launcher.
- Confirm `/v1/models` works for both control and worker endpoints from the control node.
- Confirm both endpoints advertise `max_model_len: 32768`.
- Run one minimal `/v1/chat/completions` request against both endpoints.
- Run one Qwen Code headless trivial edit against the control-node endpoint.
- Run one Qwen Code headless trivial edit against the worker-node endpoint.
- Run one isolated coding task end-to-end: Codex writes task, Qwen implements in temp worktree, tests run, Codex reviews/applies diff.

## Assumptions

- V100 SXM2 is the default target because it is more available and already practical for Qwen2.5-Coder-32B.
- Qwen2.5-Coder-32B on 4x V100 should use the 32k context target first; if startup fails with KV-cache capacity errors, retry the launcher with `MAX_MODEL_LEN=16384`.
- Qwen3-Coder is out of scope for v1 due to likely GPU/resource constraints.
- Login-node to compute-node networking is optional. The real target is control-node to worker-node communication inside one Slurm allocation.
