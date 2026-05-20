# Qwen2.5-Coder vLLM Tool-Call Findings

Date: 2026-05-20

Environment:

- Slurm job: `6928156`
- Nodes: `d1012,d1022`
- Tested from: `d1012`
- Endpoint: `http://127.0.0.1:8011/v1`
- Model: `Qwen/Qwen2.5-Coder-32B-Instruct`
- vLLM parser flags: `--enable-auto-tool-choice --tool-call-parser hermes`

## Result

The vLLM server is capable of returning real OpenAI `message.tool_calls`.

The failure mode is request/prompt/model behavior when tool choice is left to `auto`, especially through Qwen Code:

- Qwen Code sends OpenAI `tools`.
- Qwen Code does not send `tool_choice`.
- Qwen Code's system prompt includes examples using `<tool_call><function=...><parameter=...>`.
- Qwen2.5-Coder follows that text format and vLLM returns it as `message.content`.
- Because `message.tool_calls` is empty, Qwen Code never executes the apparent call.

## Direct API Probes

`tool_choice: "auto"` reproduced the issue outside Qwen Code:

```json
{
  "content": "<tools>\n{\"name\": \"get_time\", \"arguments\": {\"location\": \"New York\"}}\n</tools>",
  "tool_calls": []
}
```

Forcing a named function worked:

```json
{
  "content": "",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "get_time",
        "arguments": "{\"location\": \"New York\"}"
      }
    }
  ]
}
```

`tool_choice: "required"` also worked for the same simple tool.

Adding a stricter system instruction did not reliably work by itself; one test returned only raw JSON in `content` with no `<tool_call>` tags and no parsed `tool_calls`.

## Qwen Code Logged Request

Debug log directory:

```text
/projects/aclab/liu.shu/model-cache/tmp/qwen-openai-debug-tooltest
```

The captured request included:

- `model`
- `messages`
- `stream: true`
- `stream_options.include_usage: true`
- `tools`
- `max_tokens`

It did not include `tool_choice`.

The captured response content was:

```text
Sure, I'll list the files in the current directory using the `glob` tool.

<function=glob>
<parameter=pattern>
*
</parameter>
</function>
```

Replaying the same captured request with these changes:

- `stream: false`
- removed `stream_options`
- added `tool_choice: "required"`
- reduced `max_tokens` to `256`

returned a valid OpenAI tool call:

```json
{
  "content": "",
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "run_shell_command",
        "arguments": "{\"command\": \"ls -1\", \"directory\": \"/projects/aclab/liu.shu/codePool/hybrid_agents\"}"
      }
    }
  ]
}
```

## Interpretation

The blocker is not endpoint connectivity and not a missing vLLM tool parser. The blocker is that Qwen2.5-Coder does not reliably choose parser-compatible Hermes tool-call syntax under Qwen Code's default OpenAI-compatible request. It does generate parseable OpenAI tool calls when the request forces a tool via `tool_choice`.

## Next Options

1. Patch or wrap Qwen Code's OpenAI request path to send `tool_choice: "required"` when tools are present.
2. Insert a local proxy between Qwen Code and vLLM that adds `tool_choice: "required"` to `/chat/completions` requests with tools.
3. Write a vLLM custom parser or response shim that converts Qwen Code-style `<function=...>` text into OpenAI `tool_calls`.
4. Test another model/tool parser combination whose default behavior under `tool_choice: "auto"` is stronger.

Option 2 is the least invasive path because it does not require editing the installed Qwen Code bundle or vLLM internals.

## Proxy Experiment Result

The local proxy in `scripts/openai_tool_choice_proxy.py` was tested against Qwen Code with:

- upstream vLLM endpoint: `http://127.0.0.1:8011/v1`
- proxy endpoint: `http://127.0.0.1:18011/v1`
- task: edit a temp `hello.txt` from `hello world` to exactly `hello proxy`

The first proxy version proved that Qwen Code will execute real OpenAI `tool_calls`: it attempted `edit`, received Qwen Code's guard error that the file must be read first, called `read_file`, then called `edit` again with the correct old string. The file was updated to `hello proxy`.

Two additional proxy behaviors were needed:

1. vLLM streaming tool-call chunks sometimes ended without a non-null `finish_reason`, causing Qwen Code to fail with `Model stream ended without a finish reason`. The proxy now inserts a final streaming chunk with `finish_reason: "tool_calls"` when tool-call deltas are present and no finish reason is emitted.
2. Forcing `tool_choice: "required"` on every turn caused post-edit loops, including unnecessary verification tool calls after the file was already updated. The proxy now stops injecting `tool_choice` after a successful mutating tool result from `edit` or `write_file`.

Clean retest result:

```text
The file `/tmp/qwen-proxy-edit-test2.t6przi/hello.txt` has been successfully updated to contain exactly: `hello proxy`.
FINAL=hello proxy
```

Proxy log shape for the clean retest:

```text
Injected tool_choice='required'
Inserted streaming finish_reason='tool_calls'
Injected tool_choice='required'
Inserted streaming finish_reason='tool_calls'
Injected tool_choice='required'
Inserted streaming finish_reason='tool_calls'
POST /v1/chat/completions 200
```

The last request had no injection, allowing Qwen Code to produce the final answer and exit.
