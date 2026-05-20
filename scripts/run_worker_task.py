#!/usr/bin/env python3
"""Run one isolated Qwen Code worker task and collect artifacts."""

import argparse
import datetime
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
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


def run(cmd, cwd=None, env=None, timeout=None):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return 124, stdout, stderr + "\nTimed out after {} seconds".format(timeout)
    return proc.returncode, stdout, stderr


def is_git_repo(path):
    code, _, _ = run(["git", "-C", path, "rev-parse", "--is-inside-work-tree"])
    return code == 0


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
    code, stdout, stderr = run(["git", "-C", path, "diff", "--no-ext-diff", "--binary"])
    if code != 0:
        raise RuntimeError("git diff failed: {}".format(stderr))
    code, stdout_untracked, stderr = run(
        ["git", "-C", path, "ls-files", "--others", "--exclude-standard"]
    )
    if code != 0:
        raise RuntimeError("git ls-files failed: {}".format(stderr))
    patches = [stdout]
    for relpath in [line for line in stdout_untracked.splitlines() if line.strip()]:
        code, file_patch, stderr = run(["git", "-C", path, "diff", "--no-index", "--", "/dev/null", relpath])
        if code not in (0, 1):
            raise RuntimeError("git diff --no-index failed for {}: {}".format(relpath, stderr))
        patches.append(file_patch)
    return "".join(patches)


def plain_diff(before, after):
    code, stdout, stderr = run(["diff", "-ruN", before, after])
    if code not in (0, 1):
        raise RuntimeError("diff failed: {}".format(stderr))
    return normalize_plain_diff(stdout, before, after)


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task")
    task_group.add_argument("--task-file")
    parser.add_argument("--repo", default=os.getcwd())
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8011/v1"))
    parser.add_argument("--model")
    parser.add_argument("--qwen-bin", default=os.environ.get("QWEN_BIN", "qwen"))
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--approval-mode", default="auto-edit", choices=["default", "auto-edit", "yolo", "plan"])
    parser.add_argument("--output-format", default="text", choices=["text", "json", "stream-json"])
    parser.add_argument("--allowed-path", action="append", default=[])
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument(
        "--artifact-root",
        default=os.path.join(tempfile.gettempdir(), "hybrid_agent_tasks"),
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model-timeout", type=int, default=20)
    args = parser.parse_args(argv)

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        print("Repo path does not exist or is not a directory: {}".format(repo), file=sys.stderr)
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

    task = read_task(args)
    write_file(os.path.join(artifact_dir, "task.md"), task)

    use_git = is_git_repo(repo)
    if use_git:
        code, stdout, stderr = run(["git", "-C", repo, "worktree", "add", "--detach", workspace, "HEAD"])
        if code != 0:
            print("git worktree failed, falling back to directory copy: {}".format(stderr), file=sys.stderr)
            use_git = False

    if not use_git:
        copy_workspace(repo, before, [os.path.abspath(args.artifact_root)])
        copy_workspace(repo, workspace, [os.path.abspath(args.artifact_root)])

    model = args.model or discover_model(args.endpoint, args.model_timeout)
    prompt = (
        task
        + "\n\nOperate only inside this working directory. "
        + "Make the requested code changes, then stop. "
        + "Do not commit changes."
    )
    command = build_qwen_command(args, model, prompt)
    env = os.environ.copy()
    for key in PROXY_ENV_VARS:
        env.pop(key, None)
    env["OPENAI_BASE_URL"] = args.endpoint.rstrip("/")
    env["OPENAI_API_KEY"] = args.openai_api_key
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"

    code, stdout, stderr = run(command, cwd=workspace, env=env, timeout=args.timeout)
    write_file(os.path.join(artifact_dir, "run.log"), stdout + "\n\n[stderr]\n" + stderr)

    test_results = []
    for test_command in args.test_command:
        test_code, test_stdout, test_stderr = run(["bash", "-lc", test_command], cwd=workspace, env=env, timeout=args.timeout)
        test_results.append({"command": test_command, "returncode": test_code})
        with open(os.path.join(artifact_dir, "test.log"), "a") as handle:
            handle.write("$ {}\n{}\n[stderr]\n{}\n".format(test_command, test_stdout, test_stderr))

    patch_text = git_diff(workspace) if use_git else plain_diff(before, workspace)
    write_file(os.path.join(artifact_dir, "diff.patch"), patch_text)

    changed_paths = changed_paths_from_patch(patch_text)
    disallowed = [path for path in changed_paths if not allowed_path(path, args.allowed_path)]
    ok = code == 0 and not disallowed and all(result["returncode"] == 0 for result in test_results)

    summary = {
        "ok": ok,
        "artifact_dir": artifact_dir,
        "workspace": workspace,
        "repo": repo,
        "endpoint": args.endpoint.rstrip("/"),
        "model": model,
        "qwen_returncode": code,
        "used_git_worktree": use_git,
        "changed_paths": changed_paths,
        "disallowed_paths": disallowed,
        "tests": test_results,
    }
    write_file(os.path.join(artifact_dir, "summary.json"), json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
