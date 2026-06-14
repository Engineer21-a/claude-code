"""PyInstaller runtime hook: enforce offline mode before anything imports.

Runs before the frozen app's own code, so the HuggingFace/Transformers stack is
pinned offline even if something imports it eagerly. Mirrors app.main.enforce_offline.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
