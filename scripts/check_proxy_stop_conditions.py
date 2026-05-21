#!/usr/bin/env python3
"""Deterministic checks for openai_tool_choice_proxy stop conditions."""

import importlib.util
import os
import sys


def load_proxy_module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proxy_path = os.path.join(script_dir, "openai_tool_choice_proxy.py")
    spec = importlib.util.spec_from_file_location("openai_tool_choice_proxy", proxy_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tool_payload(name, content, arguments="{}"):
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": content,
            },
        ]
    }


def assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError("{}: expected {}, got {}".format(name, expected, actual))
    print("ok - {}".format(name))


def main():
    proxy = load_proxy_module()
    stop_tools = proxy.DEFAULT_STOP_AFTER_TOOLS

    cases = [
        (
            "edit success stops injection",
            tool_payload("edit", "File updated successfully."),
            True,
        ),
        (
            "edit create success stops injection",
            tool_payload("edit", "Created new file: /tmp/example.md with provided content."),
            True,
        ),
        (
            "write_file success stops injection",
            tool_payload("write_file", "File written successfully."),
            True,
        ),
        (
            "mutating run_shell_command success stops injection",
            tool_payload(
                "run_shell_command",
                "Command completed successfully.",
                '{"command":"mkdir -p docs && printf ok > docs/proxy.md"}',
            ),
            True,
        ),
        (
            "read-only run_shell_command does not stop injection",
            tool_payload(
                "run_shell_command",
                "docs/proxy.md",
                '{"command":"ls docs"}',
            ),
            False,
        ),
        (
            "failed mutating run_shell_command does not stop injection",
            tool_payload(
                "run_shell_command",
                "Command failed with exit code: 1",
                '{"command":"printf ok > docs/proxy.md"}',
            ),
            False,
        ),
        (
            "malformed run_shell_command arguments do not stop injection",
            tool_payload(
                "run_shell_command",
                "Command completed successfully.",
                "{not-json",
            ),
            False,
        ),
    ]

    for name, payload, expected in cases:
        actual = proxy.has_successful_stop_tool_result(payload, stop_tools)
        assert_equal(name, actual, expected)

    injection_payload = {
        "tools": [{"type": "function", "function": {"name": "edit"}}],
        "messages": cases[2][1]["messages"],
    }
    assert_equal(
        "mutating shell history disables required tool_choice injection",
        proxy.should_inject_tool_choice(
            "/v1/chat/completions",
            injection_payload,
            "required",
            stop_tools,
        ),
        False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
