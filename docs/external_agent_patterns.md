# External Coding-Agent Patterns For Hybrid Runner

Date checked: 2026-05-20

## Sources Reviewed

- Qwen Code authentication and provider docs:
  - https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/
  - https://github.com/QwenLM/qwen-code/blob/main/docs/users/configuration/model-providers.md
- OpenHands SDK workspace docs:
  - https://docs.openhands.dev/sdk/api-reference/openhands.sdk.workspace
- OpenCode provider docs:
  - https://opencode.ai/docs/providers

## Borrow

- Keep OpenAI-compatible endpoint identity explicit in every run. Qwen Code supports headless OpenAI-compatible use through `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and optional model selection; its provider config also treats the same model ID at different `baseUrl` values as distinct. Our runner should record endpoint, model, and replay command for each worker artifact.
- Treat command execution as structured data, not just logs. OpenHands exposes command results with command, stdout, stderr, exit code, and timeout status. Our runner should put the same shape in `summary.json` for Qwen and test commands, with byte counts and tails for quick lead review.
- Preserve an artifact boundary around workspaces. OpenHands separates local and remote workspaces, and our Slurm setup needs the same practical boundary: worker edits happen in an isolated git worktree or copied workspace, with `diff.patch`, logs, metadata, and summary as the contract.
- Require real tool-call compatibility for coding clients. OpenCode's provider guidance calls out that non-native models must support tool use for file operations, terminal commands, and code editing. For Qwen2.5-Coder on vLLM, that means the tool-choice proxy remains part of the viable worker path until the model/runtime combination reliably emits parsed OpenAI tool calls without it.

## Avoid

- Avoid adopting a full external agent framework for orchestration. OpenHands and OpenCode solve broader product problems; this repo needs a narrow Slurm/vLLM/Qwen Code worker runner that Codex can review and integrate.
- Avoid implicit provider state. Project or home config can change under Qwen Code. Worker artifacts should carry enough metadata to replay a run without guessing which endpoint, model, approval mode, or path gate was used.
- Avoid automatic patch application without a review gate. Worker output should be treated as a proposed patch until path allow-list checks, process/test status, and apply checks pass.
- Avoid treating Qwen multi-worker dispatch as the next orchestration step. The current runner should stay focused on single-worker reliability while Codex guidance defines when and how the lead dispatches sub-agents.

## Already Handled

- Isolated workspaces are already created through `git worktree add --detach` when possible, with directory-copy fallback.
- Proxy variables are stripped from Qwen Code worker runs, which matches the local endpoint behavior needed on Explorer.
- The worker-node tool-call path is unblocked after setting node-local `OUTLINES_CACHE_DIR`.
- Single-worker runner hardening, the review/apply gate, the proxy startup helper, and JSONL run registry records are in place for the current operational contract.

## Next Protocol Step

Codex-led hybrid dispatch should be specified as an operating protocol before adding any Qwen multi-worker CLI dispatcher:

- The lead owns architecture, decomposition, integration, verification, and the final answer.
- Delegated tasks are labeled `blocking`, `integration`, or `background`.
- Dispatch packets include lane, ownership, expected output, concurrency warning, and the lead-review integration rule.
- Safe integration requires explicit `Hybrid mode` authorization, ownership compliance, non-overlapping diffs, passing or understood checks, and lead review.
