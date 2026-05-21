#!/usr/bin/env python3
"""Run one isolated Qwen Code worker task and collect artifacts."""

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request


DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "target",
    ".venv",
    "venv",
}

PROXY_ENV_VARS = [
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "ALL_PROXY",
    "npm_config_proxy",
    "npm_config_http_proxy",
    "npm_config_https_proxy",
    "NPM_CONFIG_PROXY",
    "NPM_CONFIG_HTTP_PROXY",
    "NPM_CONFIG_HTTPS_PROXY",
]


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_RUN_REGISTRY_NAME = "runs.jsonl"


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def run(cmd, cwd=None, env=None, timeout=None):
    started_at = utc_now()
    start_monotonic = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        start_new_session=True,
    )
    timed_out = False
    timeout_grace_seconds = None
    termination_signal = None
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        timeout_grace_seconds = 10
        termination_signal = "SIGTERM"
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=timeout_grace_seconds)
        except subprocess.TimeoutExpired:
            termination_signal = "SIGKILL"
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        return {
            "command": cmd,
            "cwd": cwd,
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr + "\nTimed out after {} seconds".format(timeout),
            "timeout_seconds": timeout,
            "timeout_occurred": timed_out,
            "timeout_grace_seconds": timeout_grace_seconds,
            "termination_signal": termination_signal,
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_seconds": round(time.monotonic() - start_monotonic, 3),
            "process_group": True,
        }
    return {
        "command": cmd,
        "cwd": cwd,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timeout_seconds": timeout,
        "timeout_occurred": timed_out,
        "timeout_grace_seconds": timeout_grace_seconds,
        "termination_signal": termination_signal,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(time.monotonic() - start_monotonic, 3),
        "process_group": True,
    }


def git_has_uncommitted_tracked_changes(path):
    unstaged = run(["git", "-C", path, "diff", "--quiet", "--no-ext-diff"])
    staged = run(["git", "-C", path, "diff", "--cached", "--quiet", "--no-ext-diff"])
    return unstaged["returncode"] != 0 or staged["returncode"] != 0


def command_result_summary(result):
    return {
        "command": shell_join(result["command"]),
        "cwd": result["cwd"],
        "returncode": result["returncode"],
        "timeout_seconds": result["timeout_seconds"],
        "timeout_occurred": result["timeout_occurred"],
        "timeout_grace_seconds": result["timeout_grace_seconds"],
        "termination_signal": result["termination_signal"],
        "started_at": result["started_at"],
        "ended_at": result["ended_at"],
        "duration_seconds": result["duration_seconds"],
        "process_group": result["process_group"],
        "stdout_bytes": len(result["stdout"].encode("utf-8", errors="replace")),
        "stderr_bytes": len(result["stderr"].encode("utf-8", errors="replace")),
        "stdout_tail": result["stdout"][-2000:],
        "stderr_tail": result["stderr"][-2000:],
    }


def shell_join(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def is_git_repo(path):
    result = run(["git", "-C", path, "rev-parse", "--is-inside-work-tree"])
    return result["returncode"] == 0


def copy_workspace(src, dst, extra_exclude_roots=None):
    src = os.path.abspath(src)
    extra_exclude_roots = [os.path.abspath(path) for path in (extra_exclude_roots or [])]

    def ignore(directory, names):
        ignored = set()
        for name in names:
            if name in DEFAULT_EXCLUDES:
                ignored.add(name)
                continue
            candidate = os.path.abspath(os.path.join(directory, name))
            for root in extra_exclude_roots:
                if candidate == root or candidate.startswith(root.rstrip("/") + "/"):
                    ignored.add(name)
                    break
        return ignored

    shutil.copytree(src, dst, ignore=ignore)


def read_task(args):
    if args.task:
        return args.task
    with open(args.task_file, "r") as handle:
        return handle.read()


def discover_model(endpoint, timeout):
    endpoint = endpoint.rstrip("/")
    req = urllib.request.Request(endpoint + "/models")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError("No model id returned from {}".format(endpoint + "/models"))
    return data[0]["id"]


def git_diff(path):
    result = run(["git", "-C", path, "diff", "--no-ext-diff", "--binary"])
    if result["returncode"] != 0:
        raise RuntimeError("git diff failed: {}".format(result["stderr"]))
    untracked_result = run(
        ["git", "-C", path, "ls-files", "--others", "--exclude-standard"]
    )
    if untracked_result["returncode"] != 0:
        raise RuntimeError("git ls-files failed: {}".format(untracked_result["stderr"]))
    patches = [result["stdout"]]
    for relpath in [line for line in untracked_result["stdout"].splitlines() if line.strip()]:
        file_result = run(["git", "-C", path, "diff", "--no-index", "--", "/dev/null", relpath])
        if file_result["returncode"] not in (0, 1):
            raise RuntimeError("git diff --no-index failed for {}: {}".format(relpath, file_result["stderr"]))
        patches.append(file_result["stdout"])
    return "".join(patches)


def plain_diff(before, after):
    result = run(["diff", "-ruN", before, after])
    if result["returncode"] not in (0, 1):
        raise RuntimeError("diff failed: {}".format(result["stderr"]))
    return normalize_plain_diff(result["stdout"], before, after)


def normalize_plain_diff(patch_text, before, after):
    before = os.path.abspath(before).rstrip("/") + "/"
    after = os.path.abspath(after).rstrip("/") + "/"
    normalized = []
    for line in patch_text.splitlines():
        if line.startswith("diff -ruN "):
            parts = line.split(" ")
            if len(parts) >= 4:
                left = parts[-2].replace(before, "a/", 1).replace(after, "b/", 1)
                right = parts[-1].replace(before, "a/", 1).replace(after, "b/", 1)
                normalized.append("diff -ruN {} {}".format(left, right))
                continue
        if line.startswith("--- "):
            normalized.append(line.replace("--- " + before, "--- a/", 1).replace("--- " + after, "--- b/", 1))
        elif line.startswith("+++ "):
            normalized.append(line.replace("+++ " + before, "+++ a/", 1).replace("+++ " + after, "+++ b/", 1))
        else:
            normalized.append(line)
    if patch_text.endswith("\n"):
        return "\n".join(normalized) + "\n"
    return "\n".join(normalized)


def changed_paths_from_patch(patch_text):
    paths = set()
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.add(clean_diff_path(parts[2]))
                paths.add(clean_diff_path(parts[3]))
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            paths.add(clean_diff_path(line[6:]))
        elif line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:]
            if raw != "/dev/null":
                paths.add(clean_diff_path(raw))
    return sorted(path for path in paths if path and path != "/dev/null")


def clean_diff_path(path):
    path = path.split("\t", 1)[0]
    if path.endswith(" 00:00:00.000000000 +0000"):
        path = path.rsplit(" ", 2)[0]
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def allowed_path(path, patterns):
    if not patterns:
        return True
    normalized = path.lstrip("./")
    for pattern in patterns:
        clean = pattern.lstrip("./")
        if fnmatch.fnmatch(normalized, clean) or normalized.startswith(clean.rstrip("/") + "/"):
            return True
    return False


def write_file(path, text):
    with open(path, "w") as handle:
        handle.write(text)


def read_file(path):
    with open(path, "r") as handle:
        return handle.read()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def build_qwen_command(args, model, prompt):
    command = [
        args.qwen_bin,
        "--bare",
        "--approval-mode",
        args.approval_mode,
        "--openai-base-url",
        args.endpoint.rstrip("/"),
        "--openai-api-key",
        args.openai_api_key,
        "--auth-type",
        "openai",
        "--model",
        model,
        "--prompt",
        prompt,
    ]
    if args.output_format:
        command.extend(["--output-format", args.output_format])
    return command


def resolve_qwen_max_output_tokens(args, env):
    if args.qwen_max_output_tokens is not None:
        return str(args.qwen_max_output_tokens), "argument"
    if env.get("QWEN_CODE_MAX_OUTPUT_TOKENS"):
        return env["QWEN_CODE_MAX_OUTPUT_TOKENS"], "environment"
    return "1024", "runner_default"


def build_replay_command(args, script_path):
    command = [
        script_path,
        "--repo",
        os.path.abspath(args.repo),
    ]
    if args.task:
        command.extend(
            [
                "--endpoint",
                args.endpoint.rstrip("/"),
                "--qwen-bin",
                args.qwen_bin,
                "--approval-mode",
                args.approval_mode,
                "--output-format",
                args.output_format,
                "--timeout",
                str(args.timeout),
                "--model-timeout",
                str(args.model_timeout),
                "--qwen-max-output-tokens",
                str(args.effective_qwen_max_output_tokens),
            ]
        )
        if args.model:
            command.extend(["--model", args.model])
        command.extend(["--task", args.task])
    elif args.task_file:
        command.extend(
            [
                "--endpoint",
                args.endpoint.rstrip("/"),
                "--qwen-bin",
                args.qwen_bin,
                "--approval-mode",
                args.approval_mode,
                "--output-format",
                args.output_format,
                "--timeout",
                str(args.timeout),
                "--model-timeout",
                str(args.model_timeout),
                "--qwen-max-output-tokens",
                str(args.effective_qwen_max_output_tokens),
            ]
        )
        if args.model:
            command.extend(["--model", args.model])
        command.extend(["--task-file", args.task_file])
    elif args.review_patch:
        command.extend(["--review-patch", os.path.abspath(args.review_patch)])
    for allowed_path_pattern in args.allowed_path:
        command.extend(["--allowed-path", allowed_path_pattern])
    for test_command in args.test_command:
        command.extend(["--test-command", test_command])
    if args.apply:
        command.append("--apply")
    if args.allow_dirty_apply:
        command.append("--allow-dirty-apply")
    return "OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY} " + shell_join(command)


def classify_failure(
    qwen_result,
    test_results,
    disallowed_paths,
    apply_result,
    patch_text,
    setup_error=None,
):
    if setup_error:
        return "setup_error"
    if qwen_result and qwen_result["timeout_occurred"]:
        return "qwen_timeout"
    if qwen_result and qwen_result["returncode"] != 0:
        return "qwen_failed"
    if disallowed_paths:
        return "patch_disallowed_paths"
    if any(result["returncode"] != 0 for result in test_results):
        return "tests_failed"
    if patch_text == "":
        return "empty_patch"
    if apply_result and apply_result.get("returncode") != 0:
        return apply_result.get("failure_class") or "apply_failed"
    return None


def failure_reasons(
    failure_class,
    qwen_result,
    test_results,
    disallowed_paths,
    apply_result,
    patch_text,
    setup_error=None,
):
    reasons = []
    if setup_error:
        reasons.append(str(setup_error))
    if patch_text == "":
        reasons.append("Patch is empty.")
    if qwen_result and qwen_result["timeout_occurred"]:
        reasons.append("Qwen Code timed out after {} seconds and the runner terminated its process group.".format(qwen_result["timeout_seconds"]))
    elif qwen_result and qwen_result["returncode"] != 0:
        reasons.append("Qwen Code exited with return code {}.".format(qwen_result["returncode"]))
    if disallowed_paths:
        reasons.append("Patch changes paths outside --allowed-path: {}".format(", ".join(disallowed_paths)))
    failed_tests = [result["command"] for result in test_results if result["returncode"] != 0]
    if failed_tests:
        reasons.append("Test commands failed: {}".format("; ".join(failed_tests)))
    if apply_result and apply_result.get("returncode") != 0:
        apply_error = apply_result.get("stderr") or apply_result.get("stderr_tail") or "unknown error"
        reasons.append("Patch apply failed: {}".format(apply_error.strip()))
    if failure_class is None:
        reasons.append("All review gates passed.")
    return reasons


def review_patch(patch_text, allowed_patterns):
    changed_paths = changed_paths_from_patch(patch_text)
    disallowed = [path for path in changed_paths if not allowed_path(path, allowed_patterns)]
    return changed_paths, disallowed


def blocked_apply_result(failure_class, stderr):
    return {
        "returncode": 1,
        "failure_class": failure_class,
        "stderr": stderr,
    }


def apply_patch_to_repo(repo, patch_text, artifact_dir, allow_dirty_apply=False):
    if patch_text == "":
        return blocked_apply_result("empty_patch", "Refused to apply an empty patch.")
    if is_git_repo(repo) and not allow_dirty_apply and git_has_uncommitted_tracked_changes(repo):
        return blocked_apply_result(
            "dirty_repo",
            "Refused to apply patch because --repo has uncommitted tracked changes. Pass --allow-dirty-apply to override.",
        )
    patch_path = os.path.join(artifact_dir, "diff.patch")
    check_result = run(["git", "-C", repo, "apply", "--check", patch_path])
    if check_result["returncode"] != 0:
        summary = command_result_summary(check_result)
        summary["failure_class"] = "apply_check_conflict"
        return summary
    apply_result = run(["git", "-C", repo, "apply", patch_path])
    summary = command_result_summary(apply_result)
    if summary["returncode"] != 0:
        summary["failure_class"] = "apply_failed"
    return summary


def write_json(path, payload):
    write_file(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_run_registry(path, summary, replay_command):
    if not path:
        return
    record = {
        "timestamp": utc_now(),
        "artifact_dir": summary.get("artifact_dir"),
        "repo": summary.get("repo"),
        "endpoint": summary.get("endpoint"),
        "model": summary.get("model"),
        "mode": summary.get("mode"),
        "ok": summary.get("ok"),
        "failure_class": summary.get("failure_class"),
        "changed_paths": summary.get("changed_paths") or [],
        "patch_sha256": summary.get("patch_sha256"),
        "apply_requested": summary.get("apply_requested"),
        "apply_result": summary.get("apply_result"),
        "replay_command": replay_command,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task")
    task_group.add_argument("--task-file")
    task_group.add_argument("--review-patch", help="Review an existing patch and optionally apply it.")
    parser.add_argument("--repo", default=os.getcwd())
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8011/v1"))
    parser.add_argument("--model")
    parser.add_argument("--qwen-bin", default=os.environ.get("QWEN_BIN", "qwen"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--approval-mode", default="yolo", choices=["default", "auto-edit", "yolo", "plan"])
    parser.add_argument("--output-format", default="text", choices=["text", "json", "stream-json"])
    parser.add_argument("--allowed-path", action="append", default=[])
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="Apply the generated/reviewed patch to --repo after all review gates pass.")
    parser.add_argument(
        "--allow-dirty-apply",
        action="store_true",
        help="Allow --apply when --repo has uncommitted tracked changes.",
    )
    parser.add_argument(
        "--artifact-root",
        default=os.path.join(tempfile.gettempdir(), "hybrid_agent_tasks"),
    )
    parser.add_argument(
        "--run-registry",
        help="Append one JSONL run record here. Defaults to ARTIFACT_ROOT/runs.jsonl.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model-timeout", type=int, default=20)
    parser.add_argument(
        "--qwen-max-output-tokens",
        type=int,
        help="Set QWEN_CODE_MAX_OUTPUT_TOKENS for the Qwen Code subprocess. Defaults to 1024 unless the environment already sets it.",
    )
    args = parser.parse_args(argv)
    base_env = os.environ.copy()
    (
        args.effective_qwen_max_output_tokens,
        qwen_max_output_tokens_source,
    ) = resolve_qwen_max_output_tokens(args, base_env)

    repo = os.path.abspath(args.repo)
    args.artifact_root = os.path.abspath(args.artifact_root)
    if args.run_registry is None:
        args.run_registry = os.path.join(args.artifact_root, DEFAULT_RUN_REGISTRY_NAME)
    if not os.path.isdir(repo):
        print("Repo path does not exist or is not a directory: {}".format(repo), file=sys.stderr)
        return 2
    if not (args.task or args.task_file or args.review_patch):
        print("One of --task, --task-file, or --review-patch is required.", file=sys.stderr)
        return 2

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = os.path.abspath(os.path.join(args.artifact_root, timestamp))
    suffix = 1
    while os.path.exists(artifact_dir):
        artifact_dir = os.path.abspath(os.path.join(args.artifact_root, "{}-{}".format(timestamp, suffix)))
        suffix += 1
    os.makedirs(artifact_dir)
    workspace = os.path.join(artifact_dir, "workspace")
    before = os.path.join(artifact_dir, "before")
    script_path = os.path.abspath(sys.argv[0])
    replay_command = build_replay_command(args, script_path)

    if args.review_patch:
        patch_text = read_file(args.review_patch)
        write_file(os.path.join(artifact_dir, "diff.patch"), patch_text)
        changed_paths, disallowed = review_patch(patch_text, args.allowed_path)
        apply_result = None
        if args.apply and not disallowed and patch_text != "":
            apply_result = apply_patch_to_repo(
                repo,
                patch_text,
                artifact_dir,
                allow_dirty_apply=args.allow_dirty_apply,
            )
        elif args.apply and disallowed:
            apply_result = blocked_apply_result(
                "patch_disallowed_paths",
                "Refused to apply patch with disallowed paths.",
            )
        elif args.apply and patch_text == "":
            apply_result = blocked_apply_result("empty_patch", "Refused to apply an empty patch.")
        failure_class = classify_failure(None, [], disallowed, apply_result, patch_text)
        ok = failure_class is None
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "ok": ok,
            "mode": "review_patch",
            "failure_class": failure_class,
            "failure_reasons": failure_reasons(failure_class, None, [], disallowed, apply_result, patch_text),
            "artifact_dir": artifact_dir,
            "workspace": None,
            "repo": repo,
            "endpoint": None,
            "model": None,
            "qwen": None,
            "used_git_worktree": None,
            "changed_paths": changed_paths,
            "disallowed_paths": disallowed,
            "allowed_path_patterns": args.allowed_path,
            "tests": [],
            "patch_sha256": sha256_text(patch_text),
            "apply_requested": args.apply,
            "apply_result": apply_result,
            "replay_command": replay_command,
            "run_registry": args.run_registry,
        }
        write_json(os.path.join(artifact_dir, "summary.json"), summary)
        append_run_registry(args.run_registry, summary, replay_command)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if ok else 1

    task = read_task(args)
    write_file(os.path.join(artifact_dir, "task.md"), task)
    metadata = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "mode": "run_worker",
        "created_at": utc_now(),
        "repo": repo,
        "endpoint": args.endpoint.rstrip("/"),
        "allowed_path_patterns": args.allowed_path,
        "test_commands": args.test_command,
        "timeout_seconds": args.timeout,
        "model_timeout_seconds": args.model_timeout,
        "qwen_max_output_tokens": args.effective_qwen_max_output_tokens,
        "qwen_max_output_tokens_source": qwen_max_output_tokens_source,
        "approval_mode": args.approval_mode,
        "output_format": args.output_format,
        "apply_requested": args.apply,
        "replay_command": replay_command,
    }

    use_git = is_git_repo(repo)
    if use_git:
        worktree_result = run(["git", "-C", repo, "worktree", "add", "--detach", workspace, "HEAD"])
        metadata["worktree_add"] = command_result_summary(worktree_result)
        if worktree_result["returncode"] != 0:
            print("git worktree failed, falling back to directory copy: {}".format(worktree_result["stderr"]), file=sys.stderr)
            use_git = False

    if not use_git:
        copy_workspace(repo, before, [os.path.abspath(args.artifact_root)])
        copy_workspace(repo, workspace, [os.path.abspath(args.artifact_root)])

    try:
        model = args.model or discover_model(args.endpoint, args.model_timeout)
    except Exception as exc:
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "ok": False,
            "mode": "run_worker",
            "failure_class": "setup_error",
            "failure_reasons": failure_reasons("setup_error", None, [], [], None, None, exc),
            "artifact_dir": artifact_dir,
            "workspace": workspace,
            "repo": repo,
            "endpoint": args.endpoint.rstrip("/"),
            "model": args.model,
            "qwen_max_output_tokens": args.effective_qwen_max_output_tokens,
            "qwen_max_output_tokens_source": qwen_max_output_tokens_source,
            "qwen": None,
            "used_git_worktree": use_git,
            "changed_paths": [],
            "disallowed_paths": [],
            "allowed_path_patterns": args.allowed_path,
            "tests": [],
            "patch_sha256": None,
            "apply_requested": args.apply,
            "apply_result": None,
            "replay_command": replay_command,
            "run_registry": args.run_registry,
        }
        metadata["setup_error"] = str(exc)
        write_json(os.path.join(artifact_dir, "run_metadata.json"), metadata)
        write_json(os.path.join(artifact_dir, "summary.json"), summary)
        append_run_registry(args.run_registry, summary, replay_command)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1
    metadata["model"] = model
    prompt = (
        task
        + "\n\nOperate only inside this working directory. "
        + "Make the requested code changes, then stop. "
        + "Do not commit changes."
    )
    command = build_qwen_command(args, model, prompt)
    env = base_env.copy()
    for key in PROXY_ENV_VARS:
        env.pop(key, None)
    env["OPENAI_BASE_URL"] = args.endpoint.rstrip("/")
    env["OPENAI_API_KEY"] = args.openai_api_key
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env["QWEN_CODE_MAX_OUTPUT_TOKENS"] = args.effective_qwen_max_output_tokens

    redacted_command = list(command)
    if "--openai-api-key" in redacted_command:
        key_index = redacted_command.index("--openai-api-key") + 1
        if key_index < len(redacted_command):
            redacted_command[key_index] = "<redacted>"
    metadata["qwen_command"] = shell_join(redacted_command)
    write_json(os.path.join(artifact_dir, "run_metadata.json"), metadata)

    qwen_result = run(command, cwd=workspace, env=env, timeout=args.timeout)
    write_file(os.path.join(artifact_dir, "run.log"), qwen_result["stdout"] + "\n\n[stderr]\n" + qwen_result["stderr"])

    test_results = []
    for test_command in args.test_command:
        test_result = run(["bash", "-lc", test_command], cwd=workspace, env=env, timeout=args.timeout)
        test_results.append(command_result_summary(test_result))
        with open(os.path.join(artifact_dir, "test.log"), "a") as handle:
            handle.write("$ {}\n{}\n[stderr]\n{}\n".format(test_command, test_result["stdout"], test_result["stderr"]))

    patch_text = git_diff(workspace) if use_git else plain_diff(before, workspace)
    write_file(os.path.join(artifact_dir, "diff.patch"), patch_text)

    changed_paths, disallowed = review_patch(patch_text, args.allowed_path)
    apply_result = None
    if (
        args.apply
        and qwen_result["returncode"] == 0
        and not disallowed
        and patch_text != ""
        and all(result["returncode"] == 0 for result in test_results)
    ):
        apply_result = apply_patch_to_repo(
            repo,
            patch_text,
            artifact_dir,
            allow_dirty_apply=args.allow_dirty_apply,
        )
    elif args.apply:
        apply_result = blocked_apply_result(
            "review_gate_failed",
            "Refused to apply patch because one or more review gates failed.",
        )
    failure_class = classify_failure(qwen_result, test_results, disallowed, apply_result, patch_text)
    ok = failure_class is None

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "ok": ok,
        "mode": "run_worker",
        "failure_class": failure_class,
        "failure_reasons": failure_reasons(failure_class, qwen_result, test_results, disallowed, apply_result, patch_text),
        "artifact_dir": artifact_dir,
        "workspace": workspace,
        "repo": repo,
        "endpoint": args.endpoint.rstrip("/"),
        "model": model,
        "qwen_max_output_tokens": args.effective_qwen_max_output_tokens,
        "qwen_max_output_tokens_source": qwen_max_output_tokens_source,
        "qwen_returncode": qwen_result["returncode"],
        "qwen": command_result_summary(qwen_result),
        "used_git_worktree": use_git,
        "changed_paths": changed_paths,
        "disallowed_paths": disallowed,
        "allowed_path_patterns": args.allowed_path,
        "tests": test_results,
        "patch_sha256": sha256_text(patch_text),
        "apply_requested": args.apply,
        "apply_result": apply_result,
        "replay_command": replay_command,
        "run_registry": args.run_registry,
    }
    write_json(os.path.join(artifact_dir, "summary.json"), summary)
    append_run_registry(args.run_registry, summary, replay_command)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
