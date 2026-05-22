# Hybrid Worker Direct Generation

`scripts/hybrid-worker` (`scripts/hybrid_worker.py`) is the Codex-facing local helper for bounded work on
V100-hosted OpenAI-compatible vLLM endpoints. It is review-only by design: the
model never edits the real repo. The coordinator selects context, asks the model
for complete file content, writes that content in an isolated workspace, runs
optional checks there, and returns artifacts plus a patch for Codex/GPT-5.5 to
review.

Use this path before Qwen Code, OpenHands, or OpenCode for production helper
tasks. Keep full agentic tool-use stacks as separate benchmarks until they prove
reliable on file edits, path gates, timeout behavior, and tests.

## Task Shape

Good default tasks:

- Small function implementations.
- Boilerplate generation in one explicit file.
- Unit-test drafts.
- Documentation drafts.
- Small isolated rewrites across at most three explicit files.
- Log or output summarization captured into a target note.

Excluded by default:

- Cross-cutting refactors.
- Dependency migrations.
- Fragile production changes.
- Secrets, auth, or security-sensitive work.
- Ambiguous edits that require undocumented architecture judgment.

## Basic Use

```bash
scripts/hybrid-worker \
  --repo /path/to/repo \
  --endpoint http://127.0.0.1:8011/v1 \
  --task "Implement the requested small change." \
  --target-file src/example.py \
  --allowed-path "src/**" \
  --test-command "pytest tests/test_example.py"
```

For small multi-file work, list every target file explicitly:

```bash
scripts/hybrid-worker \
  --repo /path/to/repo \
  --endpoint http://127.0.0.1:8011/v1 \
  --task "Add the helper and a focused test." \
  --target-file src/example.py \
  --target-file tests/test_example.py \
  --allowed-path "src/**" \
  --allowed-path "tests/**" \
  --test-command "pytest tests/test_example.py"
```

If `--allowed-path` is omitted, the target files are the only allowed changed paths.
The worker refuses more than three target files.

Default `--output-protocol auto` uses plain complete-file output. A single target
file is one plain model call; multiple target files are generated as sequential
plain calls with the workspace updated after each successful file. Use
`--output-protocol json` only when explicit atomic one-shot multi-file JSON output
is required.

## Artifacts

Each run writes a timestamped directory under `/tmp/hybrid_worker_tasks` unless
`--artifact-root` is supplied:

- `task.md`: submitted task.
- `prompt.md`: exact model prompt.
- `selected_context.json`: target file contents supplied to the model.
- `raw_response.json`: full chat-completion response.
- `raw_responses.json`: all chat-completion responses for sequential plain runs.
- `raw_model_response.txt`: assistant message content.
- `raw_model_responses.json`: all assistant message contents for sequential plain runs.
- `generated_files/`: model-generated file content.
- `generated_files.json`: hashes and sizes of generated files.
- `workspace/`: isolated workspace containing generated changes.
- `diff.patch`: review patch from the isolated workspace.
- `test.log`: optional test output.
- `run_metadata.json`: endpoint, model, target files, replay command, and run settings.
- `summary.json`: stable machine-readable result.

The command also prints `summary.json` to stdout and appends a registry record to
`ARTIFACT_ROOT/runs.jsonl`.

## Endpoint Routing

By default, the worker checks `http://127.0.0.1:8011/v1` and, inside a two-node
Slurm allocation, auto-discovers the second node as `http://<worker>:8012/v1`.
Pass `--endpoint` for a single explicit endpoint, or pass an endpoint registry:

```json
{
  "endpoints": [
    {"name": "control", "base_url": "http://127.0.0.1:8011/v1"},
    {"name": "worker", "base_url": "http://d1011:8012/v1"}
  ]
}
```

The worker health-checks `/v1/models` before dispatch and selects the first
healthy endpoint that serves the requested model, falling back to the first
healthy endpoint if the model list does not exactly match.

## Dirty Repo Handling

If the target repo has uncommitted changes, the worker copies the current working
tree into the isolated workspace instead of creating a detached git worktree from
`HEAD`. This preserves reviewer-added tests and in-progress context while still
keeping model writes out of the real repo. Clean git repos use `git worktree add`
for faster artifact creation.

## Model Profiles

Default production profile:

```text
qwen25-7b-v100 -> Qwen/Qwen2.5-Coder-7B-Instruct
```

Benchmark and quality profiles:

```text
qwen25-14b-v100 -> Qwen/Qwen2.5-Coder-14B-Instruct
qwen3-30b-a3b-v100 -> Qwen/Qwen3-Coder-30B-A3B-Instruct
```

Override with `--model`, `--max-tokens`, or `--temperature` when a run needs a
specific served model.

## Safety Checks

Run deterministic offline checks without vLLM:

```bash
scripts/check_hybrid_worker.py
```

These cover:

- Real repo remains unchanged.
- Disallowed paths are rejected.
- Invalid model output is classified.
- Plain raw code succeeds.
- Full-response Markdown fences are stripped.
- Explanatory non-code output is rejected for code files.
- Sequential two-file plain generation writes both explicit targets.
- Explicit JSON multi-file mode still works.
- Test failures preserve artifacts.
- Unavailable endpoint is a setup failure, not a crash.
- Dirty source repos are copied so uncommitted tests participate in verification.
