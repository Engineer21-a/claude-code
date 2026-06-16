# Local AI Coding Loop on Google Colab (A100 / H100)

Run a fully **local, agentic coding loop** in Google Colab — no API keys, no
external model calls. A HuggingFace open-weight coding model is served with
**vLLM** (OpenAI-compatible API) and driven by the **OpenHands** autonomous
coding agent. You give it a task in natural language; it reads, edits, and runs
code in a workspace, iterating until the task is done.

```
   HuggingFace weights ──▶ vLLM server ──▶ OpenAI API ──▶ OpenHands agent loop ──▶ /workspace
        (model)            (port 8000)     (tool calls)     (CodeActAgent)         (your code)
```

Optimized for **quality + speed on a single 80 GB GPU**: FP8 on H100, BF16 on
A100, prefix caching for the agent's growing multi-turn prompts, and a fast MoE
default model (~3B active parameters).

## Quickstart

1. Open **`Local_AI_Coding_Loop.ipynb`** in Colab
   (`File ▸ Open notebook ▸ GitHub`, or upload it).
2. **Runtime ▸ Change runtime type ▸ A100 GPU** (or H100). You need a high-RAM
   80 GB GPU — Colab Pro+ / Enterprise.
3. Run the cells top to bottom. The notebook:
   - detects the GPU and picks FP8 vs BF16 automatically,
   - installs vLLM + OpenHands,
   - launches the model server in the background and waits until it's ready,
   - points OpenHands' **Local Runtime** at it,
   - runs your coding task in a re-runnable loop.
4. Edit the **task** cell and re-run it to iterate. Inspect changes with the
   `git diff` cell. **Save your work** (git push or download `/content/workspace`)
   before the Colab VM is reclaimed.

## Models

Default is **`Qwen/Qwen3.6-35B-A3B`** — as of mid-2026 this is OpenHands' own
first-recommended local model. The notebook is fully model-configurable; pick any
row below (or another HF model) in the config cell.

| Model | Total / Active | Context | License | Notes |
|---|---|---|---|---|
| **`Qwen/Qwen3.6-35B-A3B`** (default) | 35B / ~3B (MoE) | large | Apache-2.0 | OpenHands' recommended local default; best quality/speed on one 80 GB GPU |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 30B / ~3B (MoE) | 160K | Apache-2.0 | Coding-specialized, very fast; great fallback |
| `Qwen/Qwen3-Coder-Next` | 80B / ~3B (MoE) | 262K | Apache-2.0 | Higher ceiling; use **FP8** on H100, tight on A100 BF16 |
| `mistralai/Devstral-Small-2507` | 24B dense | 128K | Apache-2.0 | Built with All Hands AI specifically for OpenHands; lots of headroom |

**Bigger / frontier-open** models (DeepSeek-V4, GLM-5.1, Kimi K2.6, MiniMax M3)
score higher on SWE-Bench but need **FP8 + multiple GPUs** — out of single-GPU
scope. To use one, set `tensor_parallel_size` > 1 on a multi-GPU runtime.

## Precision: FP8 vs BF16

- **H100 / H200 / L40 / Ada / Blackwell** → FP8 (`--quantization fp8`): ~2×
  throughput and lower memory. Auto-selected by `helpers.detect_gpu()`.
- **A100** → BF16 (no native FP8 tensor cores). Auto-selected.

## Why OpenHands Local Runtime (not Docker)

OpenHands defaults to a Docker sandbox, but **Colab has no Docker**. We use the
**Local Runtime** (`RUNTIME=local`), which runs agent actions directly on the VM.

> ⚠️ **Security:** Local Runtime has **no sandbox** — the agent can read/modify
> any file on the VM and run arbitrary commands. This is acceptable on a
> disposable Colab VM, but **never** point it at a machine with secrets or data
> you care about.

## Troubleshooting

| Symptom | Fix |
|---|---|
| CUDA out of memory | Lower `MAX_MODEL_LEN` (e.g. 32768) and/or `GPU_MEM_UTIL` (e.g. 0.85); prefer FP8 on H100 |
| vLLM exits at launch | Check `/content/vllm_server.log`; usually OOM or a gated model (set `HF_TOKEN`) |
| Tool-call / function-calling errors | Parser mismatch — Qwen→`hermes`, Devstral→`mistral` (handled by `tool_parser_for`) |
| "Docker not available" | Make sure `RUNTIME=local` is set (the config cell does this) |
| Server never becomes ready | First load includes download + weight load + CUDA graph capture; raise `wait_for_server(timeout=...)` |
| Lost work after disconnect | Colab VMs are ephemeral — git push or download `/content/workspace` before idle timeout |

## Files

| File | Purpose |
|---|---|
| `Local_AI_Coding_Loop.ipynb` | The main Colab notebook (run this) |
| `helpers.py` | GPU detection, vLLM lifecycle, OpenHands config + headless run |
| `config.template.toml` | Optional OpenHands config reference |
| `requirements.txt` | Pinned dependencies |

## Sources (June 2026 research)

- OpenHands — Local LLMs & recommended local model: https://docs.openhands.dev/openhands/usage/llms/local-llms
- OpenHands — Local Runtime (no Docker): https://docs.openhands.dev/openhands/usage/runtimes/local
- vLLM — Tool calling / parsers: https://docs.vllm.ai/en/latest/features/tool_calling/
- Qwen3-Coder-Next (small hybrid coding models): https://qwen.ai/blog?id=qwen3-coder-next
- Best open-source LLMs for agentic coding (2026): https://www.mindstudio.ai/blog/best-open-source-llms-agentic-coding-2026
- Best open-source coding models 2026 (GLM/MiniMax/Qwen/Kimi): https://www.morphllm.com/best-open-source-coding-model-2026
