#!/usr/bin/env python3
"""OpenAI-compatible proxy that can force tool_choice for tool requests."""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

CONTENT_HEADERS = {"content-length", "content-encoding"}
DEFAULT_STOP_AFTER_TOOLS = ("edit", "write_file")


def normalize_base_url(url):
    return url.rstrip("/")


def normalize_tool_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def tool_result_looks_successful(tool_name, content):
    text = normalize_tool_content(content)
    lower = text.lower()
    if not text:
        return False
    failure_markers = (
        "error",
        "failed",
        "not been read",
        "cannot",
        "command:",
        "exit code:",
    )
    if any(marker in lower for marker in failure_markers):
        return False
    if tool_name == "edit":
        return "updated" in lower or "modified" in lower or "replaced" in lower
    if tool_name == "write_file":
        return "written" in lower or "created" in lower or "updated" in lower
    return True


def has_successful_stop_tool_result(payload, stop_after_tools):
    stop_after_tools = set(stop_after_tools or ())
    if not stop_after_tools:
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    call_names = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                name = function.get("name")
                call_id = tool_call.get("id")
                if name in stop_after_tools and call_id:
                    call_names[call_id] = name
        elif message.get("role") == "tool":
            name = call_names.get(message.get("tool_call_id"))
            if name and tool_result_looks_successful(name, message.get("content")):
                return True
    return False


def should_inject_tool_choice(path, payload, mode, stop_after_tools):
    if mode == "off":
        return False
    if not path.rstrip("/").endswith("/chat/completions"):
        return False
    if not isinstance(payload, dict):
        return False
    if "tool_choice" in payload:
        return False
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return False
    return not has_successful_stop_tool_result(payload, stop_after_tools)


def build_upstream_url(base_url, path):
    parsed = urllib.parse.urlsplit(path)
    clean_path = parsed.path
    query = ("?" + parsed.query) if parsed.query else ""
    return base_url + clean_path + query


class ToolChoiceProxyHandler(BaseHTTPRequestHandler):
    server_version = "OpenAIToolChoiceProxy/0.1"

    def log_message(self, fmt, *args):
        if self.server.quiet:
            return
        super().log_message(fmt, *args)

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def do_OPTIONS(self):
        self.forward()

    def forward(self):
        body = self.read_body()
        upstream_url = build_upstream_url(self.server.upstream, self.path)
        body = self.maybe_rewrite_body(body)
        headers = self.forward_headers(body)
        request = urllib.request.Request(
            upstream_url,
            data=body if self.command not in ("GET", "HEAD") else None,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.server.timeout) as response:
                response_body = response.read()
                response_body = self.maybe_rewrite_response(response, response_body)
                self.send_response(response.status, response.reason)
                self.copy_response_headers(response.headers, response_body)
                self.end_headers()
                if response_body:
                    self.wfile.write(response_body)
        except urllib.error.HTTPError as exc:
            detail = exc.read()
            self.send_response(exc.code, exc.reason)
            self.copy_response_headers(exc.headers, detail)
            self.end_headers()
            if detail:
                self.wfile.write(detail)
        except Exception as exc:
            self.send_response(502, "Bad Gateway")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error = {"error": "proxy upstream failure", "detail": str(exc)}
            self.wfile.write(json.dumps(error).encode("utf-8"))

    def read_body(self):
        length = self.headers.get("Content-Length")
        if not length:
            return None
        return self.rfile.read(int(length))

    def maybe_rewrite_body(self, body):
        if not body:
            return body
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            return body
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body
        if not should_inject_tool_choice(
            self.path,
            payload,
            self.server.tool_choice,
            self.server.stop_after_tools,
        ):
            return body
        payload["tool_choice"] = self.server.tool_choice
        if not self.server.quiet:
            print(
                "Injected tool_choice={!r} for {}".format(
                    self.server.tool_choice, self.path
                ),
                file=sys.stderr,
                flush=True,
            )
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def maybe_rewrite_response(self, response, body):
        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            return body
        if b"finish_reason" in body and b'"finish_reason":null' not in body:
            return body
        text = body.decode("utf-8", errors="replace")
        events = text.splitlines()
        metadata = None
        has_tool_delta = False
        has_finish = False
        insert_at = None
        for index, line in enumerate(events):
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                if insert_at is None:
                    insert_at = index
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            metadata = metadata or {
                "id": payload.get("id"),
                "object": payload.get("object", "chat.completion.chunk"),
                "created": payload.get("created"),
                "model": payload.get("model"),
            }
            choices = payload.get("choices") or []
            if not choices and "usage" in payload and insert_at is None:
                insert_at = index
            for choice in choices:
                if choice.get("finish_reason") is not None:
                    has_finish = True
                delta = choice.get("delta") or {}
                if delta.get("tool_calls"):
                    has_tool_delta = True
        if has_finish or not has_tool_delta or not metadata:
            return body
        finish_payload = {
            "id": metadata.get("id"),
            "object": metadata.get("object") or "chat.completion.chunk",
            "created": metadata.get("created"),
            "model": metadata.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
        }
        finish_line = "data: " + json.dumps(finish_payload, separators=(",", ":"))
        if insert_at is None:
            insert_at = len(events)
        events.insert(insert_at, "")
        events.insert(insert_at, finish_line)
        rewritten = "\n".join(events) + ("\n" if text.endswith("\n") else "")
        if not self.server.quiet:
            print("Inserted streaming finish_reason='tool_calls'", file=sys.stderr, flush=True)
        return rewritten.encode("utf-8")

    def forward_headers(self, body):
        headers = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower == "host":
                continue
            if lower == "content-length":
                continue
            headers[key] = value
        if body is not None:
            headers["Content-Length"] = str(len(body))
        return headers

    def copy_response_headers(self, headers, body):
        for key, value in headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in CONTENT_HEADERS:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body or b"")))


class ToolChoiceProxyServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        handler,
        upstream,
        tool_choice,
        stop_after_tools,
        timeout,
        quiet,
    ):
        super().__init__(server_address, handler)
        self.upstream = normalize_base_url(upstream)
        self.tool_choice = tool_choice
        self.stop_after_tools = stop_after_tools
        self.timeout = timeout
        self.quiet = quiet


def parse_stop_after_tools(value):
    if value is None:
        return DEFAULT_STOP_AFTER_TOOLS
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18011)
    parser.add_argument("--upstream", default="http://127.0.0.1:8011")
    parser.add_argument(
        "--tool-choice",
        default="required",
        choices=["required", "auto", "none", "off"],
        help="tool_choice value to inject when tools are present and absent. Use off to disable rewriting.",
    )
    parser.add_argument(
        "--stop-after-tool",
        default=",".join(DEFAULT_STOP_AFTER_TOOLS),
        help=(
            "Comma-separated tool names whose successful result disables later "
            "tool_choice injection for that request history. Use an empty value "
            "to force every tool request."
        ),
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    server = ToolChoiceProxyServer(
        (args.listen_host, args.listen_port),
        ToolChoiceProxyHandler,
        args.upstream,
        args.tool_choice,
        parse_stop_after_tools(args.stop_after_tool),
        args.timeout,
        args.quiet,
    )
    print(
        "OpenAI tool_choice proxy listening on http://{}:{} -> {}".format(
            args.listen_host, args.listen_port, normalize_base_url(args.upstream)
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
