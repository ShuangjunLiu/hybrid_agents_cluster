#!/usr/bin/env python3
"""Run a bounded direct-generation local worker task and collect artifacts."""

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request


SUMMARY_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_ROOT = os.path.join(tempfile.gettempdir(), "hybrid_worker_tasks")
DEFAULT_RUN_REGISTRY_NAME = "runs.jsonl"
DEFAULT_MODEL_PROFILES = {
    "qwen25-7b-v100": {
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "max_tokens": 4096,
        "temperature": 0,
    },
    "qwen25-14b-v100": {
        "model": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "max_tokens": 4096,
        "temperature": 0,
    },
    "qwen3-30b-a3b-v100": {
        "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "max_tokens": 8192,
        "temperature": 0,
    },
}
DEFAULT_ENDPOINTS = [
    {"name": "control", "base_url": "http://127.0.0.1:8011/v1"},
]
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
CODE_FILE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".lua",
    ".m",
    ".mm",
    ".php",
    ".pl",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".ts",
    ".tsx",
    ".vue",
}


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def timestamp():
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def write_text(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def run(command, cwd=None, timeout=None):
    started_at = utc_now()
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "command": shell_join(command),
            "cwd": cwd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "timeout_seconds": timeout,
            "timeout_occurred": False,
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_seconds": round(time.monotonic() - start, 3),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": shell_join(command),
            "cwd": cwd,
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr + "\nTimed out after {} seconds".format(timeout),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
            "timeout_seconds": timeout,
            "timeout_occurred": True,
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_seconds": round(time.monotonic() - start, 3),
        }


def shell_join(command):
    return " ".join(subprocess.list2cmdline([str(part)]) for part in command)


def request_json(method, url, payload=None, timeout=20, api_key="EMPTY"):
    body = None
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(api_key),
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} from {}: {}".format(exc.code, url, detail))
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach {}: {}".format(url, exc))


def normalize_base_url(url):
    return url.rstrip("/")


def load_endpoint_registry(args):
    endpoints = []
    if args.endpoint_registry:
        payload = read_json(args.endpoint_registry)
        if isinstance(payload, dict):
            endpoints = payload.get("endpoints") or []
        elif isinstance(payload, list):
            endpoints = payload
    if args.endpoint:
        endpoints.append({"name": "cli", "base_url": args.endpoint})
    if not endpoints:
        endpoints = list(DEFAULT_ENDPOINTS)
        worker_url = discover_slurm_worker_url(args.worker_port)
        if worker_url:
            endpoints.append({"name": "worker", "base_url": worker_url})
    return [
        {
            "name": item.get("name") or "endpoint",
            "base_url": normalize_base_url(item.get("base_url") or item.get("url")),
        }
        for item in endpoints
        if item.get("base_url") or item.get("url")
    ]


def discover_slurm_worker_url(default_port):
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return None
    result = run(["scontrol", "show", "hostnames", nodelist])
    if result["returncode"] != 0:
        return None
    nodes = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    if len(nodes) < 2:
        return None
    return "http://{}:{}/v1".format(nodes[1], default_port)


def healthy_endpoints(endpoints, timeout, api_key):
    healthy = []
    errors = []
    for endpoint in endpoints:
        try:
            payload = request_json(
                "GET",
                normalize_base_url(endpoint["base_url"]) + "/models",
                timeout=timeout,
                api_key=api_key,
            )
            data = payload.get("data") or []
            if not data:
                raise RuntimeError("/models returned no models")
            healthy.append(
                {
                    "name": endpoint["name"],
                    "base_url": normalize_base_url(endpoint["base_url"]),
                    "models": data,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "name": endpoint["name"],
                    "base_url": normalize_base_url(endpoint["base_url"]),
                    "error": str(exc),
                }
            )
    return healthy, errors


def choose_endpoint(healthy, requested_model):
    if not healthy:
        return None
    for endpoint in healthy:
        model_ids = [item.get("id") for item in endpoint["models"]]
        if requested_model in model_ids:
            return endpoint
    return healthy[0]


def resolve_profile(args):
    profile = dict(DEFAULT_MODEL_PROFILES.get(args.model_profile, DEFAULT_MODEL_PROFILES["qwen25-7b-v100"]))
    if args.model:
        profile["model"] = args.model
    if args.max_tokens is not None:
        profile["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        profile["temperature"] = args.temperature
    return profile


def is_git_repo(path):
    return run(["git", "-C", path, "rev-parse", "--is-inside-work-tree"])["returncode"] == 0


def git_worktree_dirty(path):
    result = run(["git", "-C", path, "status", "--porcelain"])
    return result["returncode"] != 0 or bool(result["stdout"].strip())


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
                if candidate == root or candidate.startswith(root.rstrip("/") + os.sep):
                    ignored.add(name)
                    break
        return ignored

    shutil.copytree(src, dst, ignore=ignore)


def materialize_workspace(repo, workspace, before, artifact_root):
    dirty = False
    if is_git_repo(repo):
        dirty = git_worktree_dirty(repo)
    if is_git_repo(repo) and not dirty:
        result = run(["git", "-C", repo, "worktree", "add", "--detach", workspace, "HEAD"])
        if result["returncode"] == 0:
            return True, False, result
    else:
        result = {"returncode": 1, "stderr": "dirty git repo copied" if dirty else "not a git repo"}
    copy_workspace(repo, before, [artifact_root])
    copy_workspace(repo, workspace, [artifact_root])
    return False, dirty, result


def git_diff(path):
    result = run(["git", "-C", path, "diff", "--no-ext-diff", "--binary"])
    if result["returncode"] != 0:
        raise RuntimeError("git diff failed: {}".format(result["stderr"]))
    untracked = run(["git", "-C", path, "ls-files", "--others", "--exclude-standard"])
    if untracked["returncode"] != 0:
        raise RuntimeError("git ls-files failed: {}".format(untracked["stderr"]))
    patches = [result["stdout"]]
    for relpath in [line for line in untracked["stdout"].splitlines() if line.strip()]:
        file_result = run(["git", "-C", path, "diff", "--no-index", "--", "/dev/null", relpath])
        if file_result["returncode"] not in (0, 1):
            raise RuntimeError("git diff --no-index failed for {}: {}".format(relpath, file_result["stderr"]))
        patches.append(file_result["stdout"])
    return "".join(patches)


def plain_diff(before, after):
    result = run(["diff", "-ruN", before, after])
    if result["returncode"] not in (0, 1):
        raise RuntimeError("diff failed: {}".format(result["stderr"]))
    before_prefix = os.path.abspath(before).rstrip("/") + "/"
    after_prefix = os.path.abspath(after).rstrip("/") + "/"
    lines = []
    for line in result["stdout"].splitlines():
        lines.append(
            line.replace(before_prefix, "a/", 1).replace(after_prefix, "b/", 1)
        )
    return "\n".join(lines) + ("\n" if result["stdout"].endswith("\n") else "")


def remove_test_byproducts(root):
    for directory, dirnames, filenames in os.walk(root):
        for name in list(dirnames):
            if name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                shutil.rmtree(os.path.join(directory, name), ignore_errors=True)
                dirnames.remove(name)
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                try:
                    os.remove(os.path.join(directory, name))
                except OSError:
                    pass


def clean_diff_path(path):
    path = path.split("\t", 1)[0]
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path


def changed_paths_from_patch(patch_text):
    paths = set()
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.add(clean_diff_path(parts[2]))
                paths.add(clean_diff_path(parts[3]))
        elif line.startswith("+++ b/") or line.startswith("--- a/"):
            paths.add(clean_diff_path(line[6:]))
        elif line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:]
            if raw != "/dev/null":
                paths.add(clean_diff_path(raw))
    return sorted(path for path in paths if path and path != "/dev/null")


def allowed_path(path, patterns):
    if not patterns:
        return True
    normalized = path.lstrip("./")
    for pattern in patterns:
        clean = pattern.lstrip("./")
        if fnmatch.fnmatch(normalized, clean) or normalized.startswith(clean.rstrip("/") + "/"):
            return True
    return False


def assert_safe_relpath(path):
    normalized = os.path.normpath(path)
    if os.path.isabs(path) or normalized == "." or normalized.startswith("..{}".format(os.sep)) or normalized == "..":
        raise ValueError("Unsafe output path: {}".format(path))
    return normalized


def selected_context(repo, target_files):
    files = []
    for relpath in target_files:
        safe = assert_safe_relpath(relpath)
        abspath = os.path.join(repo, safe)
        if os.path.exists(abspath):
            content = read_text(abspath)
            exists = True
        else:
            content = ""
            exists = False
        files.append({"path": safe, "exists": exists, "content": content})
    return {"files": files}


def build_prompt(task, context, mode, target_file=None):
    file_blocks = []
    for item in context["files"]:
        status = "existing" if item["exists"] else "new"
        file_blocks.append(
            "Path: {path} ({status})\n```text\n{content}\n```".format(
                path=item["path"], status=status, content=item["content"]
            )
        )
    if mode == "plain_file":
        if not target_file:
            raise ValueError("plain_file prompt requires target_file")
        instruction = (
            "Return only the complete final contents of {target_file}. "
            "Do not include the path, JSON, notes, explanations, Markdown, or code fences."
        ).format(target_file=target_file)
    elif mode == "single_file":
        instruction = (
            "Return only JSON with this exact shape: "
            '{"path": "relative/path", "content_lines": ["complete", "replacement", "file lines"], "notes": "brief notes"}. '
            "The path must be the target file path. content_lines must contain the complete final file, one line per array item, without trailing newline characters."
        )
    else:
        instruction = (
            "Return only JSON with this exact shape: "
            '{"files": [{"path": "relative/path", "content_lines": ["complete", "replacement", "file lines"]}], "notes": "brief notes"}. '
            "Return at most three files, and only files listed in the target context. content_lines must contain each complete final file, one line per array item, without trailing newline characters."
        )
    return textwrap.dedent(
        """\
        You are a bounded local code-generation helper. Codex/GPT-5.5 is the planner,
        reviewer, and final integrator. Generate reviewable file content only.

        Task:
        {task}

        Target context:
        {file_blocks}

        Rules:
        - Do not ask questions.
        - Do not edit files outside the target context.
        - Preserve unrelated content unless the task requires changing it.
        - {instruction}
        """
    ).format(task=task.strip(), file_blocks="\n\n".join(file_blocks), instruction=instruction)


def chat_completion(endpoint, model, prompt, max_tokens, temperature, timeout, api_key, output_protocol):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You return only complete file contents for bounded code generation tasks."
                    if output_protocol == "plain"
                    else "You return strict JSON for bounded code generation tasks."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if output_protocol == "json":
        payload["response_format"] = {"type": "json_object"}
    return request_json(
        "POST",
        normalize_base_url(endpoint) + "/chat/completions",
        payload=payload,
        timeout=timeout,
        api_key=api_key,
    )


def extract_message_content(response):
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("chat response contained no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("chat response contained empty message content")
    return content.strip()


def parse_model_json(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def parse_tolerant_single_file_content_lines(content, target_files):
    if len(target_files) != 1 or '"content_lines"' not in content:
        raise ValueError("model output is not recoverable as single-file content_lines")
    lines = content.splitlines()
    in_lines = False
    recovered = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not in_lines:
            if stripped.startswith('"content_lines"') and "[" in stripped:
                in_lines = True
            continue
        if stripped.startswith("]"):
            if not recovered:
                raise ValueError("content_lines array was empty")
            return {
                "path": target_files[0],
                "content_lines": recovered,
                "notes": "Recovered from non-strict JSON content_lines output.",
            }
        if not stripped:
            continue
        if stripped.endswith(","):
            stripped = stripped[:-1].rstrip()
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            stripped = stripped[1:-1]
        recovered.append(
            stripped.replace('\\"', '"')
            .replace("\\\\", "\\")
            .replace("\\t", "\t")
            .replace("\\n", "\n")
        )
    raise ValueError("content_lines array was not closed")


def is_code_file(path):
    name = os.path.basename(path)
    if name in {"Makefile", "Dockerfile", "dockerfile"}:
        return True
    return os.path.splitext(path)[1].lower() in CODE_FILE_EXTENSIONS


def strip_surrounding_markdown_fence(content):
    lines = content.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return content


def looks_like_explanatory_prose(content):
    first = ""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped:
            first = stripped.lower()
            break
    if not first:
        return False
    explanatory_prefixes = (
        "here is",
        "here's",
        "sure",
        "below is",
        "the complete",
        "this ",
        "i ",
        "note:",
        "notes:",
    )
    return first.startswith(explanatory_prefixes)


def parse_plain_file_content(content, target_file):
    stripped = content.strip()
    fenced = strip_surrounding_markdown_fence(content)
    if fenced != content:
        return fenced.rstrip("\n") + "\n"
    if is_code_file(target_file):
        if "```" in content:
            raise ValueError("plain output for {} included Markdown fencing with surrounding prose".format(target_file))
        if looks_like_explanatory_prose(content):
            raise ValueError("plain output for {} appears to include explanatory prose".format(target_file))
    return stripped.rstrip("\n") + "\n"


def generated_files_from_payload(payload, mode, target_files):
    target_set = {os.path.normpath(path) for path in target_files}
    if mode == "single_file":
        files = [payload]
    else:
        files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model JSON did not include generated files")
    if len(files) > 3:
        raise ValueError("model returned more than three files")
    normalized = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("file entry is not an object")
        path = assert_safe_relpath(item.get("path") or "")
        if "content_lines" in item:
            content_lines = item.get("content_lines")
            if not isinstance(content_lines, list) or not all(isinstance(line, str) for line in content_lines):
                raise ValueError("content_lines for {} is not a list of strings".format(path))
            content = "\n".join(content_lines)
            if content_lines:
                content += "\n"
        else:
            content = item.get("content")
        if not isinstance(content, str):
            raise ValueError("file content for {} is not a string".format(path))
        if path not in target_set:
            raise ValueError("model returned path outside target files: {}".format(path))
        normalized.append({"path": path, "content": content})
    return normalized


def write_generated_files(workspace, artifact_dir, files):
    generated_dir = os.path.join(artifact_dir, "generated_files")
    manifest = []
    for item in files:
        relpath = assert_safe_relpath(item["path"])
        workspace_path = os.path.join(workspace, relpath)
        generated_path = os.path.join(generated_dir, relpath)
        write_text(workspace_path, item["content"])
        write_text(generated_path, item["content"])
        manifest.append(
            {
                "path": relpath,
                "bytes": len(item["content"].encode("utf-8", errors="replace")),
                "sha256": sha256_text(item["content"]),
            }
        )
    write_json(os.path.join(artifact_dir, "generated_files.json"), manifest)
    return manifest


def classify_failure(setup_error, generation_error, parse_error, disallowed_paths, test_results, patch_text):
    if setup_error:
        return "setup_error"
    if generation_error:
        return "endpoint_failure"
    if parse_error:
        return "invalid_model_output"
    if disallowed_paths:
        return "patch_disallowed_paths"
    if any(result["returncode"] != 0 for result in test_results):
        return "tests_failed"
    if patch_text == "":
        return "empty_patch"
    return None


def failure_reasons(failure_class, setup_error, generation_error, parse_error, disallowed_paths, test_results, patch_text):
    if failure_class is None:
        return ["All review gates passed."]
    reasons = []
    if setup_error:
        reasons.append(str(setup_error))
    if generation_error:
        reasons.append(str(generation_error))
    if parse_error:
        reasons.append(str(parse_error))
    if disallowed_paths:
        reasons.append("Patch changes paths outside --allowed-path: {}".format(", ".join(disallowed_paths)))
    failed_tests = [result["command"] for result in test_results if result["returncode"] != 0]
    if failed_tests:
        reasons.append("Test commands failed: {}".format("; ".join(failed_tests)))
    if patch_text == "":
        reasons.append("Patch is empty.")
    return reasons


def build_replay_command(args, script_path):
    command = [
        script_path,
        "--repo",
        os.path.abspath(args.repo),
        "--task",
        args.task or "",
        "--model-profile",
        args.model_profile,
        "--artifact-root",
        os.path.abspath(args.artifact_root),
        "--timeout",
        str(args.timeout),
    ]
    if args.endpoint:
        command.extend(["--endpoint", args.endpoint])
    if args.endpoint_registry:
        command.extend(["--endpoint-registry", os.path.abspath(args.endpoint_registry)])
    if args.model:
        command.extend(["--model", args.model])
    if args.output_protocol != "auto":
        command.extend(["--output-protocol", args.output_protocol])
    for path in args.target_file:
        command.extend(["--target-file", path])
    for pattern in args.allowed_path:
        command.extend(["--allowed-path", pattern])
    for test_command in args.test_command:
        command.extend(["--test-command", test_command])
    return shell_join(command)


def append_run_registry(path, summary):
    if not path:
        return
    record = {
        "timestamp": utc_now(),
        "artifact_dir": summary.get("artifact_dir"),
        "repo": summary.get("repo"),
        "endpoint": summary.get("endpoint"),
        "model": summary.get("model"),
        "ok": summary.get("ok"),
        "failure_class": summary.get("failure_class"),
        "changed_paths": summary.get("changed_paths") or [],
        "patch_sha256": summary.get("patch_sha256"),
        "replay_command": summary.get("replay_command"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getcwd())
    parser.add_argument("--task")
    parser.add_argument("--task-file")
    parser.add_argument("--target-file", action="append", default=[])
    parser.add_argument("--allowed-path", action="append", default=[])
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument("--endpoint")
    parser.add_argument("--endpoint-registry")
    parser.add_argument("--worker-port", type=int, default=8012)
    parser.add_argument("--model-profile", default="qwen25-7b-v100", choices=sorted(DEFAULT_MODEL_PROFILES))
    parser.add_argument("--model")
    parser.add_argument("--output-protocol", default="auto", choices=("auto", "plain", "json"))
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-registry")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--health-timeout", type=int, default=20)
    args = parser.parse_args(argv)

    repo = os.path.abspath(args.repo)
    args.artifact_root = os.path.abspath(args.artifact_root)
    if args.run_registry is None:
        args.run_registry = os.path.join(args.artifact_root, DEFAULT_RUN_REGISTRY_NAME)
    if args.task_file:
        task = read_text(args.task_file)
    else:
        task = args.task
    if not task:
        print("One of --task or --task-file is required.", file=sys.stderr)
        return 2
    if not os.path.isdir(repo):
        print("Repo path does not exist or is not a directory: {}".format(repo), file=sys.stderr)
        return 2
    if not args.target_file:
        print("At least one --target-file is required for direct generation.", file=sys.stderr)
        return 2
    if len(args.target_file) > 3:
        print("Direct worker tasks are limited to at most three --target-file values.", file=sys.stderr)
        return 2

    target_files = [assert_safe_relpath(path) for path in args.target_file]
    if args.output_protocol == "plain" and len(target_files) > 3:
        print("Plain worker tasks are limited to at most three --target-file values.", file=sys.stderr)
        return 2
    effective_output_protocol = args.output_protocol
    if effective_output_protocol == "auto":
        effective_output_protocol = "plain"
    allowed_patterns = args.allowed_path or target_files
    run_id = timestamp()
    artifact_dir = os.path.join(args.artifact_root, run_id)
    suffix = 1
    while os.path.exists(artifact_dir):
        artifact_dir = os.path.join(args.artifact_root, "{}-{}".format(run_id, suffix))
        suffix += 1
    os.makedirs(artifact_dir)
    workspace = os.path.join(artifact_dir, "workspace")
    before = os.path.join(artifact_dir, "before")
    replay_command = build_replay_command(args, os.path.abspath(sys.argv[0]))
    profile = resolve_profile(args)

    setup_error = None
    generation_error = None
    parse_error = None
    response = None
    response_content = None
    generation = None
    generated_manifest = []
    patch_text = ""
    changed_paths = []
    disallowed_paths = []
    test_results = []
    used_git_worktree = None
    repo_dirty_copied = None
    endpoint_errors = []
    selected_endpoint = None
    selected_model = profile["model"]

    try:
        used_git_worktree, repo_dirty_copied, worktree_result = materialize_workspace(repo, workspace, before, args.artifact_root)
        context = selected_context(workspace, target_files)
        prompt = build_prompt(
            task,
            context,
            "plain_file" if effective_output_protocol == "plain" else ("single_file" if len(target_files) == 1 else "multi_file"),
            target_files[0] if effective_output_protocol == "plain" else None,
        )
        write_text(os.path.join(artifact_dir, "task.md"), task)
        write_json(os.path.join(artifact_dir, "selected_context.json"), context)
        write_text(os.path.join(artifact_dir, "prompt.md"), prompt)
        metadata = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "created_at": utc_now(),
            "repo": repo,
            "target_files": target_files,
            "allowed_path_patterns": allowed_patterns,
            "model_profile": args.model_profile,
            "model": selected_model,
            "output_protocol": effective_output_protocol,
            "requested_output_protocol": args.output_protocol,
            "max_tokens": profile["max_tokens"],
            "temperature": profile["temperature"],
            "test_commands": args.test_command,
            "review_only": True,
            "replay_command": replay_command,
            "used_git_worktree": used_git_worktree,
            "repo_dirty_copied": repo_dirty_copied,
            "worktree_result": worktree_result,
        }
        write_json(os.path.join(artifact_dir, "run_metadata.json"), metadata)

        healthy, endpoint_errors = healthy_endpoints(
            load_endpoint_registry(args), args.health_timeout, args.openai_api_key
        )
        selected_endpoint = choose_endpoint(healthy, selected_model)
        if not selected_endpoint:
            setup_error = "No healthy endpoint available. Endpoint errors: {}".format(endpoint_errors)
        else:
            try:
                generation_started_at = utc_now()
                generation_start = time.monotonic()
                generated_files = []
                raw_responses = []
                raw_contents = []
                if effective_output_protocol == "plain":
                    for target_file in target_files:
                        step_context = selected_context(workspace, target_files)
                        step_prompt = build_prompt(task, step_context, "plain_file", target_file)
                        write_text(
                            os.path.join(artifact_dir, "prompt.{}.md".format(target_file.replace(os.sep, "__"))),
                            step_prompt,
                        )
                        response = chat_completion(
                            selected_endpoint["base_url"],
                            selected_model,
                            step_prompt,
                            profile["max_tokens"],
                            profile["temperature"],
                            args.timeout,
                            args.openai_api_key,
                            effective_output_protocol,
                        )
                        raw_responses.append(response)
                        response_content = extract_message_content(response)
                        raw_contents.append({"path": target_file, "content": response_content})
                        generated_files.append(
                            {
                                "path": target_file,
                                "content": parse_plain_file_content(response_content, target_file),
                            }
                        )
                        generated_manifest = write_generated_files(workspace, artifact_dir, generated_files)
                else:
                    response = chat_completion(
                        selected_endpoint["base_url"],
                        selected_model,
                        prompt,
                        profile["max_tokens"],
                        profile["temperature"],
                        args.timeout,
                        args.openai_api_key,
                        effective_output_protocol,
                    )
                    raw_responses.append(response)
                    response_content = extract_message_content(response)
                    raw_contents.append({"path": None, "content": response_content})
                    try:
                        payload = parse_model_json(response_content)
                    except Exception:
                        payload = parse_tolerant_single_file_content_lines(
                            response_content,
                            target_files,
                        )
                    generated_files = generated_files_from_payload(
                        payload,
                        "single_file" if len(target_files) == 1 else "multi_file",
                        target_files,
                    )
                    generated_manifest = write_generated_files(workspace, artifact_dir, generated_files)
                generation = {
                    "started_at": generation_started_at,
                    "ended_at": utc_now(),
                    "duration_seconds": round(time.monotonic() - generation_start, 3),
                    "timeout_seconds": args.timeout,
                    "requests": len(raw_responses),
                }
                if raw_responses:
                    write_json(os.path.join(artifact_dir, "raw_response.json"), raw_responses[-1])
                    write_json(os.path.join(artifact_dir, "raw_responses.json"), raw_responses)
                if raw_contents:
                    write_text(os.path.join(artifact_dir, "raw_model_response.txt"), raw_contents[-1]["content"])
                    write_json(os.path.join(artifact_dir, "raw_model_responses.json"), raw_contents)
            except ValueError as exc:
                parse_error = exc
            except Exception as exc:
                generation_error = exc

        if not setup_error and not generation_error and not parse_error:
            for test_command in args.test_command:
                result = run(["bash", "-lc", test_command], cwd=workspace, timeout=args.timeout)
                test_results.append(result)
                with open(os.path.join(artifact_dir, "test.log"), "a", encoding="utf-8") as handle:
                    handle.write("$ {}\n{}\n[stderr]\n{}\n".format(test_command, result["stdout"], result["stderr"]))
            remove_test_byproducts(workspace)
            patch_text = git_diff(workspace) if used_git_worktree else plain_diff(before, workspace)
            write_text(os.path.join(artifact_dir, "diff.patch"), patch_text)
            changed_paths = changed_paths_from_patch(patch_text)
            disallowed_paths = [path for path in changed_paths if not allowed_path(path, allowed_patterns)]
        else:
            write_text(os.path.join(artifact_dir, "diff.patch"), "")
    except Exception as exc:
        setup_error = setup_error or exc

    failure_class = classify_failure(
        setup_error,
        generation_error,
        parse_error,
        disallowed_paths,
        test_results,
        patch_text,
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "ok": failure_class is None,
        "mode": "direct_generation_review",
        "failure_class": failure_class,
        "failure_reasons": failure_reasons(
            failure_class,
            setup_error,
            generation_error,
            parse_error,
            disallowed_paths,
            test_results,
            patch_text,
        ),
        "artifact_dir": artifact_dir,
        "workspace": workspace,
        "repo": repo,
        "endpoint": selected_endpoint["base_url"] if selected_endpoint else None,
        "endpoint_name": selected_endpoint["name"] if selected_endpoint else None,
        "endpoint_errors": endpoint_errors,
        "model_profile": args.model_profile,
        "model": selected_model,
        "output_protocol": effective_output_protocol,
        "requested_output_protocol": args.output_protocol,
        "generation": generation,
        "review_only": True,
        "used_git_worktree": used_git_worktree,
        "repo_dirty_copied": repo_dirty_copied,
        "target_files": target_files,
        "generated_files": generated_manifest,
        "changed_paths": changed_paths,
        "disallowed_paths": disallowed_paths,
        "allowed_path_patterns": allowed_patterns,
        "tests": [
            {
                key: value
                for key, value in result.items()
                if key not in ("stdout", "stderr")
            }
            for result in test_results
        ],
        "patch_sha256": sha256_text(patch_text) if patch_text else None,
        "replay_command": replay_command,
        "run_registry": args.run_registry,
    }
    write_json(os.path.join(artifact_dir, "summary.json"), summary)
    append_run_registry(args.run_registry, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
