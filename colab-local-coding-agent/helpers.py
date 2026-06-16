"""Helper utilities for the Local AI Coding Loop Colab notebook.

These keep the notebook cells short and readable. Everything here is plain
Python with no Colab-specific imports, so the module can also be unit-imported
and statically checked outside Colab.

Pipeline:  vLLM server  <->  OpenAI-compatible API  <->  OpenHands agent loop
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# GPU detection
# --------------------------------------------------------------------------- #


@dataclass
class GpuInfo:
    name: str
    memory_gb: float
    supports_fp8: bool
    recommended_quantization: str | None  # "fp8" or None (BF16)
    dtype: str  # "auto" / "bfloat16"

    @property
    def summary(self) -> str:
        quant = self.recommended_quantization or "none (BF16)"
        return (
            f"GPU: {self.name} | {self.memory_gb:.0f} GB | "
            f"FP8: {self.supports_fp8} | quantization: {quant}"
        )


def detect_gpu() -> GpuInfo:
    """Query nvidia-smi and recommend a precision strategy.

    H100 / H200 / Ada (and newer) have FP8 tensor cores -> use FP8 for ~2x
    throughput and lower memory. A100 has no native FP8 -> fall back to BF16.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return GpuInfo("unknown (no nvidia-smi)", 0.0, False, None, "auto")

    first = out.splitlines()[0]
    name, mem_mib = (p.strip() for p in first.split(","))
    memory_gb = float(mem_mib) / 1024.0

    upper = name.upper()
    # FP8-capable: Hopper (H100/H200/H20), Ada Lovelace (L4/L40/4090), Blackwell.
    supports_fp8 = any(tag in upper for tag in ("H100", "H200", "H20", "L40", "L4", "4090", "B100", "B200", "GH200"))

    if supports_fp8:
        return GpuInfo(name, memory_gb, True, "fp8", "auto")
    return GpuInfo(name, memory_gb, False, None, "bfloat16")


# --------------------------------------------------------------------------- #
# Model / tool-call-parser mapping
# --------------------------------------------------------------------------- #

# OpenHands-recommended local default (June 2026) first, then alternatives.
SUPPORTED_MODELS = (
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Coder-Next",
    "mistralai/Devstral-Small-2507",
)


def tool_parser_for(model: str) -> str:
    """Return the vLLM --tool-call-parser appropriate for the model family.

    Qwen chat templates use Hermes-style tool calls; Mistral/Devstral use the
    mistral parser. Defaults to hermes (the common case here).
    """
    lower = model.lower()
    if "devstral" in lower or "mistral" in lower:
        return "mistral"
    return "hermes"


# --------------------------------------------------------------------------- #
# vLLM server lifecycle
# --------------------------------------------------------------------------- #

_VLLM_PROC: subprocess.Popen | None = None
_VLLM_LOG = "/content/vllm_server.log"


def build_vllm_command(
    model: str,
    *,
    gpu: GpuInfo,
    port: int = 8000,
    api_key: str = "local-key",
    max_model_len: int = 65536,
    gpu_memory_utilization: float = 0.92,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """Construct the `vllm serve ...` argv, tuned for the detected GPU."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--api-key", api_key,
        "--enable-auto-tool-choice",
        "--tool-call-parser", tool_parser_for(model),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        # Enable prefix caching: agent loops resend a growing prompt, so this
        # is a big speedup for OpenHands' multi-turn trajectories.
        "--enable-prefix-caching",
    ]
    if gpu.recommended_quantization == "fp8":
        cmd += ["--quantization", "fp8", "--kv-cache-dtype", "fp8"]
    else:
        cmd += ["--dtype", gpu.dtype]
    return cmd


def start_vllm(model: str, *, gpu: GpuInfo, log_path: str = _VLLM_LOG, **kwargs) -> subprocess.Popen:
    """Launch vLLM as a background process, streaming logs to `log_path`."""
    global _VLLM_PROC
    if _VLLM_PROC is not None and _VLLM_PROC.poll() is None:
        print("vLLM already running (pid %d). Call stop_vllm() first to restart." % _VLLM_PROC.pid)
        return _VLLM_PROC

    cmd = build_vllm_command(model, gpu=gpu, **kwargs)
    print("Launching vLLM:\n  " + " ".join(cmd) + f"\n  logs -> {log_path}")
    log = open(log_path, "w")
    _VLLM_PROC = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    return _VLLM_PROC


def wait_for_server(
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "local-key",
    timeout: int = 1800,
    poll_every: int = 10,
) -> bool:
    """Poll GET {base_url}/models until the server is ready or timeout.

    First load can take a while (download + weight load + CUDA graph capture),
    hence the generous default timeout.
    """
    url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _VLLM_PROC is not None and _VLLM_PROC.poll() is not None:
            raise RuntimeError(
                f"vLLM exited early (code {_VLLM_PROC.returncode}). "
                f"Check the server log ({_VLLM_LOG})."
            )
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("vLLM server is ready ✅")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        print("  …waiting for vLLM to come up")
        time.sleep(poll_every)
    raise TimeoutError(f"vLLM not ready after {timeout}s — see {_VLLM_LOG}")


def stop_vllm() -> None:
    """Terminate the vLLM process group and free the GPU."""
    global _VLLM_PROC
    if _VLLM_PROC is None or _VLLM_PROC.poll() is not None:
        print("vLLM is not running.")
        _VLLM_PROC = None
        return
    print(f"Stopping vLLM (pid {_VLLM_PROC.pid})…")
    try:
        os.killpg(os.getpgid(_VLLM_PROC.pid), signal.SIGTERM)
        _VLLM_PROC.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(_VLLM_PROC.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    _VLLM_PROC = None
    print("vLLM stopped, GPU freed.")


# --------------------------------------------------------------------------- #
# OpenHands configuration + headless run
# --------------------------------------------------------------------------- #


def configure_openhands(
    model: str,
    *,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "local-key",
    workspace: str = "/content/workspace",
    temperature: float = 0.0,
) -> dict[str, str]:
    """Set the env vars that point OpenHands' Local Runtime at our vLLM server.

    The `openai/` prefix tells LiteLLM (used by OpenHands) to treat this as an
    OpenAI-compatible custom endpoint. RUNTIME=local avoids Docker (Colab has none).
    """
    os.makedirs(workspace, exist_ok=True)
    env = {
        "LLM_MODEL": f"openai/{model}",
        "LLM_BASE_URL": base_url,
        "LLM_API_KEY": api_key,
        "LLM_TEMPERATURE": str(temperature),
        # Local runtime = run actions directly on the VM, no Docker sandbox.
        "RUNTIME": "local",
        "SANDBOX_VOLUMES": f"{workspace}:/workspace:rw",
    }
    os.environ.update(env)
    return env


def run_openhands_task(
    task: str,
    *,
    max_iterations: int = 50,
    config_file: str | None = None,
) -> int:
    """Run one OpenHands headless trajectory (the agentic coding loop).

    Streams the agent's output live and returns the process exit code. Re-run
    with a new `task` to iterate.
    """
    cmd = [
        sys.executable, "-m", "openhands.core.main",
        "-t", task,
        "--max-iterations", str(max_iterations),
    ]
    if config_file:
        cmd += ["--config-file", config_file]
    print("Running OpenHands:\n  " + " ".join(cmd[:3]) + f" -t <task> --max-iterations {max_iterations}\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return proc.wait()
