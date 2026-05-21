#!/usr/bin/env python3
"""Deterministic dry checks for run_worker_task.py patch review/apply gates."""

import json
import os
import subprocess
import sys
import tempfile


def run(command, cwd=None, check=False):
    result = subprocess.run(
        command,
        cwd=cwd,
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


def write(path, text):
    with open(path, "w") as handle:
        handle.write(text)


def init_repo(root):
    repo = os.path.join(root, "repo")
    os.makedirs(repo)
    run(["git", "init"], cwd=repo, check=True)
    run(["git", "config", "user.email", "runner-check@example.invalid"], cwd=repo, check=True)
    run(["git", "config", "user.name", "Runner Check"], cwd=repo, check=True)
    os.makedirs(os.path.join(repo, "docs"))
    os.makedirs(os.path.join(repo, "src"))
    write(os.path.join(repo, "docs", "allowed.md"), "old\n")
    write(os.path.join(repo, "src", "conflict.txt"), "base\n")
    run(["git", "add", "."], cwd=repo, check=True)
    run(["git", "commit", "-m", "init"], cwd=repo, check=True)
    return repo


def patch_for(path, old, new):
    return """diff --git a/{0} b/{0}
--- a/{0}
+++ b/{0}
@@ -1 +1 @@
-{1}
+{2}
""".format(path, old.rstrip("\n"), new.rstrip("\n"))


def invoke_runner(script, repo, patch_path, artifact_root, *extra):
    command = [
        script,
        "--repo",
        repo,
        "--review-patch",
        patch_path,
        "--artifact-root",
        artifact_root,
        "--allowed-path",
        "docs/**",
    ]
    command.extend(extra)
    result = run(command)
    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "runner did not emit JSON: {}\nstdout:\n{}\nstderr:\n{}".format(
                exc, result.stdout, result.stderr
            )
        )
    return result, summary


def assert_case(name, condition, detail):
    if not condition:
        raise AssertionError("{}: {}".format(name, detail))
    print("ok - {}".format(name))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runner = os.path.join(script_dir, "run_worker_task.py")

    with tempfile.TemporaryDirectory(prefix="worker-runner-check.") as root:
        repo = init_repo(root)
        artifact_root = os.path.join(root, "artifacts")

        allowed_patch = os.path.join(root, "allowed.patch")
        write(allowed_patch, patch_for("docs/allowed.md", "old", "new"))
        result, summary = invoke_runner(runner, repo, allowed_patch, artifact_root)
        assert_case("allowed path review passes", result.returncode == 0 and summary["ok"], summary)
        assert_case(
            "registry receives allowed review",
            os.path.exists(os.path.join(artifact_root, "runs.jsonl")),
            "missing runs.jsonl",
        )

        disallowed_patch = os.path.join(root, "disallowed.patch")
        write(disallowed_patch, patch_for("src/conflict.txt", "base", "changed"))
        result, summary = invoke_runner(runner, repo, disallowed_patch, artifact_root)
        assert_case(
            "disallowed path review fails",
            result.returncode == 1 and summary["failure_class"] == "patch_disallowed_paths",
            summary,
        )

        empty_patch = os.path.join(root, "empty.patch")
        write(empty_patch, "")
        result, summary = invoke_runner(runner, repo, empty_patch, artifact_root)
        assert_case(
            "empty patch review fails",
            result.returncode == 1 and summary["failure_class"] == "empty_patch",
            summary,
        )

        write(os.path.join(repo, "docs", "allowed.md"), "dirty\n")
        result, summary = invoke_runner(runner, repo, allowed_patch, artifact_root, "--apply")
        assert_case(
            "dirty tracked repo refuses apply",
            result.returncode == 1 and summary["failure_class"] == "dirty_repo",
            summary,
        )
        run(["git", "checkout", "--", "docs/allowed.md"], cwd=repo, check=True)

        conflict_patch = os.path.join(root, "conflict.patch")
        write(conflict_patch, patch_for("docs/allowed.md", "missing", "new"))
        result, summary = invoke_runner(runner, repo, conflict_patch, artifact_root, "--apply")
        assert_case(
            "apply check conflict is classified",
            result.returncode == 1 and summary["failure_class"] == "apply_check_conflict",
            summary,
        )

        with open(os.path.join(artifact_root, "runs.jsonl"), "r") as handle:
            registry_records = [json.loads(line) for line in handle if line.strip()]
        assert_case(
            "registry records each dry run",
            len(registry_records) == 5,
            "expected 5 records, got {}".format(len(registry_records)),
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
