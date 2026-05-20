# Single-A100 Qwen2.5 Tool-Call Probe

Use this when the two-node V100 allocation is unavailable and we only have one
A100 on the `sharing` partition. The goal is not production coding throughput;
the goal is to quickly test whether vLLM returns real OpenAI `tool_calls` for
Qwen2.5-Coder.

## Recommended Model Order

1. `Qwen/Qwen2.5-Coder-7B-Instruct`
   - Default for the short `sharing` allocation.
   - Fits comfortably on one A100 and should load fastest.
   - Good enough to debug parser behavior.
2. `Qwen/Qwen2.5-Coder-14B-Instruct`
   - Try after the 7B probe works.
   - Keep `MAX_MODEL_LEN=8192` first.
3. `Qwen/Qwen2.5-Coder-32B-Instruct`
   - Do not use for this one-GPU 60-minute probe unless using a tested
     quantized checkpoint. The previous BF16/FP16 32B setup was built around
     tensor parallelism across 4 V100s per server.

## Start From Login Node

Preferred launcher, installed to match the existing `~/env_sh` workflow:

```bash
source ~/env_sh/sharing_a100_qwen25coder_interactive.sh
```

Repo-local copy:

```bash
cd /projects/aclab/liu.shu/codePool/hybrid_agents
source scripts/a100_sharing_qwen25coder_interactive.sh
```

For 14B:

```bash
MODEL=Qwen/Qwen2.5-Coder-14B-Instruct MAX_MODEL_LEN=8192 \
  source scripts/a100_sharing_qwen25coder_interactive.sh
```

The launcher writes resume state to:

```text
~/.current_a100_qwen25coder_job
```

Default runtime env:

```text
/projects/aclab/liu.shu/envs/qwen25-vllm092-torch27-cu126/bin/python
```

This is a clean conda env with vLLM `0.9.2`, Torch `2.7.0+cu126`, NumPy
`2.2.6`, numba `0.61.2`, Transformers `4.51.3`,
`huggingface-hub` `0.36.2`, and tokenizers `0.21.4`.

Do not let this vLLM env float to the latest Transformers release. With
Transformers `5.8.1`, vLLM `0.9.2` fails during startup with:

```text
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
```

The fix is to keep Transformers on the `4.51.x` line for this vLLM stack.

## Probe Tool Calls

Once the launcher prints that the server is ready:

```bash
cd /projects/aclab/liu.shu/codePool/hybrid_agents
scripts/probe_openai_tool_calls.py \
  --endpoint "$OPENAI_BASE_URL" \
  --tool-choice auto \
  --out /tmp/qwen25-tool-auto.json
```

Then force a named tool:

```bash
scripts/probe_openai_tool_calls.py \
  --endpoint "$OPENAI_BASE_URL" \
  --tool-choice named \
  --out /tmp/qwen25-tool-named.json
```

Interpretation:

- `has_tool_calls: true`: vLLM parsed the model output into OpenAI
  `message.tool_calls`.
- `has_tool_calls: false` with `<function=...>` in `content_prefix`: the server
  accepted tools, but model output did not match the active parser.
- If named tool choice works but auto does not, Qwen Code may need a mode that
  uses stronger `tool_choice` guidance.

## Resume Checklist

1. Check the saved job state:

```bash
cat ~/.current_a100_qwen25coder_job
```

2. If the allocation is still alive, SSH/tunnel to the printed node as needed:

```bash
ssh -N -L 8000:<node>:8000 liu.shu@explorer.neu.edu
```

3. Check logs from the saved `OUT_LOG` and `ERR_LOG` paths.
4. Rerun the two probe commands above and compare `/tmp/qwen25-tool-*.json`.

## Explorer Runtime Tips

- Keep model download and local client networking separate. Do not set
  `NO_PROXY='*'` before vLLM resolves/downloads a Hugging Face model; that can
  bypass Explorer's proxy and stall outbound HTTPS from compute nodes.
- Do not leave `NO_PROXY='*' no_proxy='*'` exported in the interactive shell
  used to launch Codex. It makes Codex bypass Explorer's proxy for
  `chatgpt.com`, which can break `codex_apps` MCP/plugin sync with direct
  outbound HTTPS timeouts. The launcher uses the Explorer proxy for model
  download, then exports local-only bypass values for `localhost`,
  `127.0.0.1`, the allocated node, and the node FQDN after `/v1/models` is
  healthy.
- For live diagnostics inside the allocation, use `--cpu-bind=none` with
  `srun --overlap` if Slurm reports `CPU binding outside of job step
  allocation`.

Example:

```bash
srun --overlap --jobid=<job_id> --nodes=1 --ntasks=1 \
  --nodelist=<node> --cpus-per-task=1 --cpu-bind=none \
  bash -lc 'hostname; nvidia-smi'
```
