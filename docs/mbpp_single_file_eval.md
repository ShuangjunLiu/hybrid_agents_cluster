# MBPP Single-File Direct Worker Pilot

This pilot evaluates `Qwen/Qwen2.5-Coder-7B-Instruct` as a constrained
single-file direct-generation worker on the official sanitized MBPP dataset.

Dataset:

- `/projects/aclab/liu.shu/datasets/mbpp/sanitized-mbpp.json`
- Source note: `/projects/aclab/liu.shu/datasets/mbpp/README.md`

Run artifacts:

- `/projects/aclab/liu.shu/model-cache/tmp/qwen_mbpp_single_file_eval/<run_id>`

## Dry Fixture Check

Verify that the generated evaluator fails against an empty `solution.py` and
passes against the MBPP reference code:

```bash
scripts/run_mbpp_single_file_eval.py --run-id dry-fixture-check --limit 1 --dry-run-fixture
```

## Prepare Mini Repos Only

Create the first 25 selected tasks without calling Qwen:

```bash
scripts/run_mbpp_single_file_eval.py --run-id prepare-25 --prepare-only
```

Each task repo contains:

- `solution.py`: the only file Qwen may edit.
- `check_solution.py`: coordinator-generated evaluator.
- `mbpp_task.json`: task metadata for inspection.

## Live 25-Task Pilot

Run inside a validated Qwen 7B allocation with the OpenAI-compatible vLLM endpoint
available at `http://127.0.0.1:8011/v1`:

```bash
scripts/validate_vllm_endpoints.py --timeout 60 --expected-max-model-len 16384 --json

scripts/run_mbpp_single_file_eval.py --run-id mbpp-25
```

Defaults select sanitized MBPP tasks with `task_id` in `11..510`, sorted by
`task_id`, limited to the first 25. Each task calls `scripts/hybrid_worker.py`
with:

- `--target-file solution.py`
- `--allowed-path solution.py`
- `--test-command "python check_solution.py"`
- `--timeout 300`

## Aggregate Results

```bash
scripts/aggregate_mbpp_single_file_eval.py \
  /projects/aclab/liu.shu/model-cache/tmp/qwen_mbpp_single_file_eval/mbpp-25
```

Use `--json` for machine-readable output. The aggregate reports total tasks,
pass count, pass rate, timeout count, empty patch count, endpoint/model failure
count, tests-failed count, median runtime, and per-task artifact paths.

Acceptance for the 25-task pilot:

- no runner crashes
- artifacts are replayable
- pass rate at least 80% on the first 25 sanitized tasks
- median generation latency below 10 seconds on the 7B baseline
- zero disallowed-path edits
- failures are classifiable from `summary.json`
