#!/usr/bin/env python3
"""Deterministic checks for the direct-generation hybrid worker."""

import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


class MockOpenAIHandler(BaseHTTPRequestHandler):
    responses = []

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json(
                {
                    "object": "list",
                    "data": [{"id": MODEL, "object": "model", "max_model_len": 16384}],
                }
            )
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        content = self.responses.pop(0)
        self.send_json(
            {
                "id": "mock",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": content}}],
            }
        )

    def send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def init_repo(root, files):
    repo = os.path.join(root, "repo")
    os.makedirs(repo)
    run(["git", "init", "-q"], cwd=repo, check=True)
    run(["git", "config", "user.email", "hybrid-worker-check@example.invalid"], cwd=repo, check=True)
    run(["git", "config", "user.name", "Hybrid Worker Check"], cwd=repo, check=True)
    for path, content in files.items():
        write(os.path.join(repo, path), content)
    run(["git", "add", "."], cwd=repo, check=True)
    run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def start_server(responses):
    MockOpenAIHandler.responses = list(responses)
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, "http://127.0.0.1:{}/v1".format(server.server_address[1])


def invoke_worker(script, repo, artifact_root, endpoint, *extra):
    command = [
        sys.executable,
        script,
        "--repo",
        repo,
        "--endpoint",
        endpoint,
        "--artifact-root",
        artifact_root,
        "--task",
        "Make the requested bounded change.",
        "--timeout",
        "30",
        "--health-timeout",
        "5",
    ]
    command.extend(extra)
    result = run(command)
    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "worker did not emit JSON: {}\nstdout:\n{}\nstderr:\n{}".format(
                exc, result.stdout, result.stderr
            )
        )
    return result, summary


def assert_case(name, condition, detail):
    if not condition:
        raise AssertionError("{}: {}".format(name, detail))
    print("ok - {}".format(name))


def main():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hybrid_worker.py")
    with tempfile.TemporaryDirectory(prefix="hybrid-worker-check.") as root:
        server, endpoint = start_server(
            [
                "def answer():\n    return 42\n",
                "```python\ndef answer():\n    return 43\n```",
                "Here is the complete file:\ndef answer():\n    return 44\n",
                "alpha\n",
                "beta\n",
                json.dumps(
                    {
                        "files": [
                            {"path": "src/one.py", "content": "ONE = 1\n"},
                            {"path": "src/two.py", "content": "TWO = 2\n"},
                        ],
                        "notes": "implemented",
                    }
                ),
                json.dumps(
                    {
                        "files": [
                            {"path": "docs/allowed.md", "content": "allowed\n"},
                            {"path": "src/secret.txt", "content": "secret\n"},
                        ],
                        "notes": "mixed paths",
                    }
                ),
                json.dumps(
                    {
                        "path": "solution.py",
                        "content": "def value():\n    return 0\n",
                        "notes": "will fail tests",
                    }
                ),
                "def ready_order_ids(orders):\n"
                "    ready = [order for order in orders if order.get('status') == 'ready']\n"
                "    return [order['id'] for order in sorted(ready, key=lambda order: (-order.get('priority', 0), order.get('due'), order['id']))]\n",
            ]
        )
        try:
            repo = init_repo(root, {"src/answer.py": "def answer():\n    return 0\n"})
            artifacts = os.path.join(root, "artifacts1")
            result, summary = invoke_worker(
                script,
                repo,
                artifacts,
                endpoint,
                "--target-file",
                "src/answer.py",
                "--allowed-path",
                "src/**",
                "--test-command",
                "python -m py_compile src/answer.py",
            )
            assert_case("single-file generation passes", result.returncode == 0 and summary["ok"], summary)
            assert_case("single-file auto uses plain protocol", summary["output_protocol"] == "plain", summary)
            assert_case(
                "real repo is unchanged",
                read(os.path.join(repo, "src/answer.py")) == "def answer():\n    return 0\n",
                read(os.path.join(repo, "src/answer.py")),
            )
            assert_case("patch records target path", summary["changed_paths"] == ["src/answer.py"], summary)

            repo = init_repo(os.path.join(root, "fenced"), {"src/answer.py": "def answer():\n    return 0\n"})
            result, summary = invoke_worker(
                script,
                repo,
                os.path.join(root, "artifacts2"),
                endpoint,
                "--target-file",
                "src/answer.py",
                "--test-command",
                "python -m py_compile src/answer.py",
            )
            assert_case("single-file fenced code is stripped", result.returncode == 0 and summary["ok"], summary)
            assert_case(
                "fenced output patch contains raw code",
                "return 43" in read(os.path.join(summary["artifact_dir"], "diff.patch"))
                and "```" not in read(os.path.join(summary["artifact_dir"], "diff.patch")),
                read(os.path.join(summary["artifact_dir"], "diff.patch")),
            )

            repo = init_repo(os.path.join(root, "invalid"), {"src/answer.py": "old\n"})
            result, summary = invoke_worker(
                script,
                repo,
                os.path.join(root, "artifacts3"),
                endpoint,
                "--target-file",
                "src/answer.py",
            )
            assert_case(
                "explanatory non-code output is rejected",
                result.returncode == 1 and summary["failure_class"] == "invalid_model_output",
                summary,
            )

            repo = init_repo(os.path.join(root, "sequential"), {"docs/a.txt": "old a\n", "docs/b.txt": "old b\n"})
            result, summary = invoke_worker(
                script,
                repo,
                os.path.join(root, "artifacts4"),
                endpoint,
                "--target-file",
                "docs/a.txt",
                "--target-file",
                "docs/b.txt",
            )
            assert_case("sequential two-file plain generation passes", result.returncode == 0 and summary["ok"], summary)
            assert_case("sequential generation makes two requests", summary["generation"]["requests"] == 2, summary)
            assert_case("sequential generation writes both targets", summary["changed_paths"] == ["docs/a.txt", "docs/b.txt"], summary)

            repo = init_repo(os.path.join(root, "jsonok"), {"src/one.py": "ONE = 0\n", "src/two.py": "TWO = 0\n"})
            result, summary = invoke_worker(
                script,
                repo,
                os.path.join(root, "artifacts5"),
                endpoint,
                "--output-protocol",
                "json",
                "--target-file",
                "src/one.py",
                "--target-file",
                "src/two.py",
                "--test-command",
                "python -m py_compile src/one.py src/two.py",
            )
            assert_case("explicit JSON multi-file mode works", result.returncode == 0 and summary["ok"], summary)
            assert_case("explicit JSON mode records protocol", summary["output_protocol"] == "json", summary)

            repo = init_repo(
                os.path.join(root, "disallowed"),
                {"docs/allowed.md": "old\n", "src/secret.txt": "old\n"},
            )
            result, summary = invoke_worker(
                script,
                repo,
                os.path.join(root, "artifacts6"),
                endpoint,
                "--output-protocol",
                "json",
                "--target-file",
                "docs/allowed.md",
                "--target-file",
                "src/secret.txt",
                "--allowed-path",
                "docs/**",
            )
            assert_case(
                "disallowed generated path is rejected",
                result.returncode == 1 and summary["failure_class"] == "patch_disallowed_paths",
                summary,
            )

            repo = init_repo(os.path.join(root, "testfail"), {"solution.py": "def value():\n    return 1\n"})
            result, summary = invoke_worker(
                script,
                repo,
                os.path.join(root, "artifacts7"),
                endpoint,
                "--output-protocol",
                "json",
                "--target-file",
                "solution.py",
                "--test-command",
                "python -c 'import solution; raise SystemExit(solution.value() != 1)'",
            )
            assert_case(
                "test failure is classified",
                result.returncode == 1 and summary["failure_class"] == "tests_failed",
                summary,
            )
            assert_case(
                "test failure still leaves repo unchanged",
                read(os.path.join(repo, "solution.py")) == "def value():\n    return 1\n",
                read(os.path.join(repo, "solution.py")),
            )

            repo = init_repo(
                os.path.join(root, "dirty"),
                {
                    "src/order_utils.py": "def ready_order_ids(orders):\n    raise NotImplementedError\n",
                    "tests/test_order_utils.py": "import unittest\n\nfrom src.order_utils import ready_order_ids\n\n\nclass OrderUtilsTest(unittest.TestCase):\n    def test_placeholder(self):\n        self.assertTrue(callable(ready_order_ids))\n",
                },
            )
            write(
                os.path.join(repo, "tests", "test_order_utils.py"),
                read(os.path.join(repo, "tests", "test_order_utils.py"))
                + "\nclass DirtyHiddenTest(unittest.TestCase):\n"
                + "    def test_dirty_hidden_due_tie(self):\n"
                + "        self.assertEqual(ready_order_ids([\n"
                + "        {'id': 'undated', 'status': 'ready', 'priority': 1},\n"
                + "        {'id': 'dated', 'status': 'ready', 'priority': 1, 'due': '2026-05-21'},\n"
                + "        ]), ['dated', 'undated'])\n",
            )
            result, summary = invoke_worker(
                script,
                repo,
                os.path.join(root, "artifacts8"),
                endpoint,
                "--target-file",
                "src/order_utils.py",
                "--allowed-path",
                "src/order_utils.py",
                "--test-command",
                "python -m unittest discover -s tests",
            )
            assert_case(
                "dirty repo is copied into workspace",
                summary["repo_dirty_copied"] is True and summary["used_git_worktree"] is False,
                summary,
            )
            assert_case(
                "dirty uncommitted tests participate",
                result.returncode == 1 and summary["failure_class"] == "tests_failed",
                summary,
            )
        finally:
            server.shutdown()

        repo = init_repo(os.path.join(root, "unavailable"), {"src/answer.py": "old\n"})
        result, summary = invoke_worker(
            script,
            repo,
            os.path.join(root, "artifacts9"),
            "http://127.0.0.1:9/v1",
            "--target-file",
            "src/answer.py",
            "--health-timeout",
            "1",
        )
        assert_case(
            "endpoint unavailable is setup failure",
            result.returncode == 1 and summary["failure_class"] == "setup_error",
            summary,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
