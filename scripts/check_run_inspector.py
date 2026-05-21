#!/usr/bin/env python3
"""Deterministic checks for inspect_worker_runs.py."""

import json
import os
import subprocess
import sys
import tempfile


def run(command, check=False):
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "{} failed:\nstdout:\n{}\nstderr:\n{}".format(
                " ".join(command), result.stdout, result.stderr
            )
        )
    return result


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path, payload):
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def assert_case(name, condition, detail):
    if not condition:
        raise AssertionError("{}: {}".format(name, detail))
    print("ok - {}".format(name))


def make_artifact(root, name, ok, failure_class, changed_paths):
    artifact_dir = os.path.join(root, name)
    os.makedirs(artifact_dir)
    summary = {
        "artifact_dir": artifact_dir,
        "changed_paths": changed_paths,
        "disallowed_paths": [] if ok else ["secrets.txt"],
        "failure_class": failure_class,
        "failure_reasons": ["All review gates passed."] if ok else ["Patch changes paths outside --allowed-path: secrets.txt"],
        "mode": "run_worker",
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "ok": ok,
        "patch_sha256": "abc123" if ok else "def456",
        "qwen": {
            "duration_seconds": 1.25,
            "returncode": 0,
            "timeout_occurred": False,
        },
        "replay_command": "scripts/run_worker_task.py --repo /tmp/repo",
        "tests": [
            {
                "duration_seconds": 0.5,
                "returncode": 0 if ok else 1,
                "timeout_occurred": False,
            }
        ],
    }
    write_json(os.path.join(artifact_dir, "summary.json"), summary)
    return artifact_dir


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inspector = os.path.join(script_dir, "inspect_worker_runs.py")

    with tempfile.TemporaryDirectory(prefix="run-inspector-check.") as root:
        ok_dir = make_artifact(root, "20260521T010101Z", True, None, ["docs/ok.md"])
        failed_dir = make_artifact(root, "20260521T020202Z", False, "patch_disallowed_paths", ["secrets.txt"])
        registry = os.path.join(root, "runs.jsonl")
        append_jsonl(
            registry,
            {
                "timestamp": "2026-05-21T01:01:01Z",
                "artifact_dir": ok_dir,
                "ok": True,
                "failure_class": None,
                "mode": "run_worker",
                "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "changed_paths": ["docs/ok.md"],
            },
        )
        append_jsonl(
            registry,
            {
                "timestamp": "2026-05-21T02:02:02Z",
                "artifact_dir": failed_dir,
                "ok": False,
                "failure_class": "patch_disallowed_paths",
                "mode": "review_patch",
                "model": None,
                "changed_paths": ["secrets.txt"],
            },
        )

        result = run([inspector, "--artifact-root", root, "--limit", "2"], check=True)
        assert_case("human list includes ok run", "docs/ok.md" in result.stdout and "ok" in result.stdout, result.stdout)
        assert_case("human list includes failed run", "patch_disallowed_paths" in result.stdout, result.stdout)

        result = run([inspector, "--registry", registry, "--json"], check=True)
        records = json.loads(result.stdout)
        assert_case("json list returns two records", len(records) == 2, records)
        assert_case("json list is compact", set(records[0]) == {"artifact_dir", "changed_paths", "failure_class", "mode", "model", "ok", "status", "timestamp"}, records[0])

        result = run([inspector, "--registry", registry, "--status", "failed", "--json"], check=True)
        failed_records = json.loads(result.stdout)
        assert_case("failed filter returns one record", len(failed_records) == 1 and failed_records[0]["status"] == "failed", failed_records)

        result = run([inspector, "--show", failed_dir], check=True)
        assert_case("show output includes replay command", "replay_command:" in result.stdout, result.stdout)
        assert_case("show output includes disallowed paths", "secrets.txt" in result.stdout, result.stdout)

        result = run([inspector, "--show", ok_dir, "--json"], check=True)
        summary = json.loads(result.stdout)
        assert_case("show json emits summary", summary["artifact_dir"] == ok_dir and summary["ok"] is True, summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
