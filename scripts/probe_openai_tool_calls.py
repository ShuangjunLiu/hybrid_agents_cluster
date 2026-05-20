#!/usr/bin/env python3
"""Probe whether an OpenAI-compatible endpoint returns parsed tool_calls."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


PROXY_ENV_VARS = [
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "ALL_PROXY",
]


def request_json(method, url, payload=None, timeout=60):
    data = None
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} from {}: {}".format(exc.code, url, detail))
    except urllib.error.URLError as exc:
        raise RuntimeError("Could not reach {}: {}".format(url, exc))


def discover_model(endpoint, timeout):
    payload = request_json("GET", endpoint.rstrip("/") + "/models", timeout=timeout)
    data = payload.get("data") or []
    if not data or not data[0].get("id"):
        raise RuntimeError("No model id returned from /models")
    return data[0]["id"]


def build_payload(model, tool_choice):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the current workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to read.",
                        }
                    },
                    "required": ["path"],
                },
            },
        }
    ]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a coding assistant. Use tools when needed.",
            },
            {
                "role": "user",
                "content": "Read the file README.md. Use the read_file tool.",
            },
        ],
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0,
        "max_tokens": 256,
    }
    if tool_choice == "named":
        payload["tool_choice"] = {"type": "function", "function": {"name": "read_file"}}
    return payload


def summarize(response):
    choices = response.get("choices") or []
    message = (choices[0].get("message") if choices else {}) or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    return {
        "has_tool_calls": bool(tool_calls),
        "tool_calls_count": len(tool_calls),
        "content_prefix": (content or "")[:500],
        "tool_calls": tool_calls,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model")
    parser.add_argument("--tool-choice", default="auto", choices=["auto", "required", "named"])
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out")
    parser.add_argument("--keep-proxy", action="store_true")
    args = parser.parse_args(argv)

    if not args.keep_proxy:
        for key in PROXY_ENV_VARS:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

    endpoint = args.endpoint.rstrip("/")
    model = args.model or discover_model(endpoint, args.timeout)
    payload = build_payload(model, args.tool_choice)
    response = request_json("POST", endpoint + "/chat/completions", payload, args.timeout)
    result = {
        "endpoint": endpoint,
        "model": model,
        "tool_choice": args.tool_choice,
        "summary": summarize(response),
        "request": payload,
        "response": response,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0 if result["summary"]["has_tool_calls"] else 1


if __name__ == "__main__":
    sys.exit(main())
