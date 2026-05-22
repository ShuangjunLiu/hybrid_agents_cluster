#!/usr/bin/env python3
"""Run MBPP sanitized tasks as isolated single-file direct worker jobs."""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import textwrap
import urllib.request


DEFAULT_DATASET = "/projects/aclab/liu.shu/datasets/mbpp/sanitized-mbpp.json"
DEFAULT_ARTIFACT_ROOT = "/projects/aclab/liu.shu/model-cache/tmp/qwen_mbpp_single_file_eval"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ENDPOINT = "http://127.0.0.1:8011/v1"


def utc_timestamp():
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path, text):
    with open(path, "w") as handle:
        handle.write(text)


def run(command, cwd=None, timeout=None):
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout,
            stderr=stderr + "\nTimed out after {} seconds".format(timeout),
        )


def load_tasks(dataset_path, task_id_min, task_id_max, limit):
    if dataset_path.endswith(".csv"):
        tasks = []
        with open(dataset_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                task = {
                    "task_id": int(row["task_id"]),
                    "prompt": row["prompt"],
                    "code": row["code"],
                    "test_imports": json.loads(row["test_imports"]) if row["test_imports"] else [],
                    "test_list": json.loads(row["test_list"]) if row["test_list"] else [],
                    "source_file": row["source_file"],
                }
                tasks.append(task)
    else:
        tasks = read_json(dataset_path)

    selected = [
        task
        for task in tasks
        if task_id_min <= int(task["task_id"]) <= task_id_max
    ]
    selected.sort(key=lambda task: int(task["task_id"]))
    if limit is not None:
        selected = selected[:limit]
    return selected


def solution_stub(task):
    prompt = task["prompt"].strip()
    return (
        '"""MBPP task {task_id}.\n\n'
        "{prompt}\n"
        '"""\n\n'
        "# Implement the requested function here. The evaluator imports this file.\n"
    ).format(task_id=task["task_id"], prompt=prompt.replace('"""', '\\"\\"\\"'))


def check_solution_source(task):
    test_imports = task.get("test_imports") or []
    test_list = task.get("test_list") or []
    body_lines = [
        "import os",
        "import sys",
        "",
        "sys.path.insert(0, os.path.dirname(__file__))",
        "from solution import *  # noqa: F401,F403",
        "",
    ]
    body_lines.extend(test_imports)
    if test_imports:
        body_lines.append("")
    body_lines.extend(test_list)
    body_lines.append("")
    return "\n".join(body_lines)


def initialize_git_repo(repo_dir):
    run(["git", "init", "-q"], cwd=repo_dir)
    run(["git", "add", "."], cwd=repo_dir)
    result = run(
        [
            "git",
            "-c",
            "user.name=MBPP Harness",
            "-c",
            "user.email=mbpp-harness@example.invalid",
            "commit",
            "-q",
            "-m",
            "Initial MBPP fixture",
        ],
        cwd=repo_dir,
    )
    if result.returncode != 0:
        raise RuntimeError("git commit failed in {}: {}".format(repo_dir, result.stderr))


def materialize_repo(task, repo_dir, use_reference=False):
    os.makedirs(repo_dir, exist_ok=True)
    write_text(os.path.join(repo_dir, ".gitignore"), "__pycache__/\n*.py[cod]\n")
    write_text(
        os.path.join(repo_dir, "solution.py"),
        task["code"].rstrip() + "\n" if use_reference else solution_stub(task),
    )
    write_text(os.path.join(repo_dir, "check_solution.py"), check_solution_source(task))
    write_json(
        os.path.join(repo_dir, "mbpp_task.json"),
        {
            "source_file": task.get("source_file"),
            "task_id": task["task_id"],
            "prompt": task["prompt"],
            "test_imports": task.get("test_imports") or [],
            "test_list": task.get("test_list") or [],
            "reference_code_present": use_reference,
        },
    )
    initialize_git_repo(repo_dir)


def validate_fixture_pair(task, run_dir):
    task_id = int(task["task_id"])
    stub_repo = os.path.join(run_dir, "dry_run", "task_{:03d}_stub".format(task_id))
    reference_repo = os.path.join(run_dir, "dry_run", "task_{:03d}_reference".format(task_id))
    materialize_repo(task, stub_repo, use_reference=False)
    materialize_repo(task, reference_repo, use_reference=True)

    stub = run([sys.executable, "check_solution.py"], cwd=stub_repo, timeout=30)
    reference = run([sys.executable, "check_solution.py"], cwd=reference_repo, timeout=30)
    return {
        "task_id": task_id,
        "stub_repo": stub_repo,
        "reference_repo": reference_repo,
        "stub_returncode": stub.returncode,
        "reference_returncode": reference.returncode,
        "stub_stdout_tail": stub.stdout[-1000:],
        "stub_stderr_tail": stub.stderr[-1000:],
        "reference_stdout_tail": reference.stdout[-1000:],
        "reference_stderr_tail": reference.stderr[-1000:],
        "ok": stub.returncode != 0 and reference.returncode == 0,
    }


def discover_model(endpoint, timeout):
    with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data:
        raise RuntimeError("No models returned from {}".format(endpoint))
    return data[0].get("id"), data[0]


def recursive_find_key(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = recursive_find_key(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = recursive_find_key(child, key)
            if found is not None:
                return found
    return None


def require_endpoint(args):
    model_id, model_record = discover_model(args.endpoint, args.model_timeout)
    if args.expected_model and model_id != args.expected_model:
        raise RuntimeError(
            "Endpoint model mismatch: got {}, expected {}".format(
                model_id, args.expected_model
            )
        )
    max_model_len = recursive_find_key(model_record, "max_model_len")
    if args.expected_max_model_len is not None:
        if max_model_len is None:
            raise RuntimeError("Endpoint /models did not expose max_model_len")
        if int(max_model_len) != args.expected_max_model_len:
            raise RuntimeError(
                "Endpoint max_model_len mismatch: got {}, expected {}".format(
                    max_model_len, args.expected_max_model_len
                )
            )
    return model_id, max_model_len


def build_worker_prompt(task):
    tests = "\n".join(task.get("test_list") or [])
    return textwrap.dedent(
        """\
        Implement the MBPP task in solution.py only.

        Task {task_id}: {prompt}

        The coordinator will run `python check_solution.py`.
        Do not edit check_solution.py, mbpp_task.json, or any other file.

        Tests:
        {tests}
        """
    ).format(task_id=task["task_id"], prompt=task["prompt"].strip(), tests=tests)


def run_worker_for_task(args, task, repo_dir, worker_root):
    command = [
        os.path.join(args.repo_root, "scripts", "hybrid_worker.py"),
        "--repo",
        repo_dir,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--timeout",
        str(args.timeout),
        "--health-timeout",
        str(args.model_timeout),
        "--max-tokens",
        str(args.max_tokens),
        "--artifact-root",
        worker_root,
        "--target-file",
        "solution.py",
        "--allowed-path",
        "solution.py",
        "--test-command",
        "python check_solution.py",
        "--task",
        build_worker_prompt(task),
    ]
    result = run(command, timeout=args.timeout + 120)
    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError:
        summary = {
            "ok": False,
            "failure_class": "runner_output_error",
            "artifact_dir": None,
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-2000:],
        }
    if summary.get("artifact_dir"):
        summary_path = os.path.join(summary["artifact_dir"], "summary.json")
        if os.path.exists(summary_path):
            saved = read_json(summary_path)
            saved["benchmark"] = {
                "name": "mbpp_sanitized_single_file",
                "task_id": int(task["task_id"]),
                "repo_dir": repo_dir,
            }
            write_json(summary_path, saved)
            summary = saved
    return {
        "task_id": int(task["task_id"]),
        "repo_dir": repo_dir,
        "runner_returncode": result.returncode,
        "runner_stdout_tail": result.stdout[-2000:],
        "runner_stderr_tail": result.stderr[-2000:],
        "summary": summary,
        "artifact_dir": summary.get("artifact_dir"),
        "ok": summary.get("ok") is True,
        "failure_class": summary.get("failure_class"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default=utc_timestamp())
    parser.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-max-model-len", type=int, default=16384)
    parser.add_argument("--task-id-min", type=int, default=11)
    parser.add_argument("--task-id-max", type=int, default=510)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--model-timeout", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--prepare-only", action="store_true", help="Create mini repos but do not call Qwen.")
    parser.add_argument("--dry-run-fixture", action="store_true", help="Verify one stub fixture fails and the reference fixture passes.")
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(os.path.join(args.artifact_root, args.run_id))
    if os.path.exists(run_dir):
        raise SystemExit("Run directory already exists: {}".format(run_dir))
    os.makedirs(run_dir)

    tasks = load_tasks(args.dataset, args.task_id_min, args.task_id_max, args.limit)
    if not tasks:
        raise SystemExit("No MBPP tasks selected.")

    manifest = {
        "schema_version": 1,
        "benchmark": "mbpp_sanitized_single_file",
        "dataset": os.path.abspath(args.dataset),
        "run_id": args.run_id,
        "run_dir": run_dir,
        "artifact_root": os.path.abspath(args.artifact_root),
        "task_id_min": args.task_id_min,
        "task_id_max": args.task_id_max,
        "limit": args.limit,
        "selected_task_ids": [int(task["task_id"]) for task in tasks],
        "endpoint": args.endpoint,
        "model": args.model,
        "timeout_seconds": args.timeout,
        "prepare_only": args.prepare_only,
        "dry_run_fixture": args.dry_run_fixture,
        "results": [],
    }

    if args.dry_run_fixture:
        dry_result = validate_fixture_pair(tasks[0], run_dir)
        manifest["dry_run_result"] = dry_result
        write_json(os.path.join(run_dir, "mbpp_run_manifest.json"), manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if dry_result["ok"] else 1

    endpoint_result = None
    if not args.prepare_only:
        model_id, max_model_len = require_endpoint(args)
        endpoint_result = {"model": model_id, "max_model_len": max_model_len}
    manifest["endpoint_validation"] = endpoint_result

    repos_root = os.path.join(run_dir, "repos")
    worker_root = os.path.join(run_dir, "worker_artifacts")
    os.makedirs(repos_root)
    os.makedirs(worker_root)

    for task in tasks:
        task_id = int(task["task_id"])
        repo_dir = os.path.join(repos_root, "task_{:03d}".format(task_id))
        materialize_repo(task, repo_dir, use_reference=False)
        if args.prepare_only:
            record = {
                "task_id": task_id,
                "repo_dir": repo_dir,
                "artifact_dir": None,
                "ok": None,
                "failure_class": None,
            }
        else:
            record = run_worker_for_task(
                args,
                task,
                repo_dir,
                os.path.join(worker_root, "task_{:03d}".format(task_id)),
            )
        manifest["results"].append(record)
        write_json(os.path.join(run_dir, "mbpp_run_manifest.json"), manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if args.prepare_only or all(item["ok"] for item in manifest["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
