#!/usr/bin/env python3
"""Validate vLLM OpenAI-compatible endpoints for the hybrid coding setup."""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


def normalize_base_url(url):
    return url.rstrip("/")


def request_json(method, url, payload=None, timeout=20):
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} from {}: {}".format(exc.code, url, detail))
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach {}: {}".format(url, exc))


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


def discover_worker_url(default_port):
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return None
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    nodes = [line.strip() for line in output.splitlines() if line.strip()]
    if len(nodes) < 2:
        return None
    return "http://{}:{}/v1".format(nodes[1], default_port)


def validate_endpoint(name, base_url, args):
    base_url = normalize_base_url(base_url)
    models = request_json("GET", base_url + "/models", timeout=args.timeout)
    data = models.get("data") or []
    if not data:
        raise RuntimeError("{} returned no models from /v1/models".format(name))

    model_id = args.model or data[0].get("id")
    max_model_len = recursive_find_key(data[0], "max_model_len")
    if args.expected_max_model_len is not None:
        if max_model_len is None:
            raise RuntimeError(
                "{} /models did not expose max_model_len; response keys: {}".format(
                    name, sorted(data[0].keys())
                )
            )
        if int(max_model_len) != args.expected_max_model_len:
            raise RuntimeError(
                "{} max_model_len mismatch: got {}, expected {}".format(
                    name, max_model_len, args.expected_max_model_len
                )
            )

    chat_summary = None
    if not args.skip_chat:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 8,
            "temperature": 0,
        }
        chat = request_json("POST", base_url + "/chat/completions", payload, args.timeout)
        choices = chat.get("choices") or []
        if not choices:
            raise RuntimeError("{} returned no chat choices".format(name))
        message = choices[0].get("message") or {}
        chat_summary = (message.get("content") or "").strip()

    return {
        "name": name,
        "base_url": base_url,
        "model": model_id,
        "max_model_len": max_model_len,
        "chat": chat_summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--worker-url")
    parser.add_argument("--worker-port", type=int, default=8012)
    parser.add_argument("--model")
    parser.add_argument("--expected-max-model-len", type=int)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    endpoints = [("control", args.control_url)]
    worker_url = args.worker_url or discover_worker_url(args.worker_port)
    if worker_url:
        endpoints.append(("worker", worker_url))

    results = []
    errors = []
    for name, url in endpoints:
        try:
            results.append(validate_endpoint(name, url, args))
        except Exception as exc:
            errors.append({"name": name, "base_url": url, "error": str(exc)})

    if args.json:
        print(json.dumps({"ok": not errors, "results": results, "errors": errors}, indent=2))
    else:
        for result in results:
            print(
                "OK {name}: {base_url} model={model} max_model_len={max_model_len} chat={chat!r}".format(
                    **result
                )
            )
        for error in errors:
            print("FAIL {name}: {base_url} {error}".format(**error), file=sys.stderr)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
