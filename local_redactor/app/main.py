"""LocalRedactor entrypoint.

Sets the offline environment variables BEFORE any ML library is imported
(Hard Invariant: no network calls ever, unless the user opts into an update
check), then launches the Qt GUI.
"""
from __future__ import annotations

import os
import sys


def enforce_offline() -> None:
    """Pin the HuggingFace / Transformers stack to offline mode.

    Must run before `gliner2`, `transformers`, or `huggingface_hub` import,
    so we set it at process start.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv: list[str] | None = None) -> int:
    enforce_offline()

    # Import lazily so `app.main.enforce_offline` and the CLI smoke-test work
    # even in environments without PySide6 installed (e.g. headless CI).
    try:
        from app.gui.main_window import run_app
    except ImportError as exc:  # pragma: no cover - depends on GUI extras
        print(
            "LocalRedactor GUI requires PySide6. Install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            f"(import error: {exc})",
            file=sys.stderr,
        )
        return 1

    return run_app(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
