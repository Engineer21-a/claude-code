# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for a standalone LocalRedactor.exe (Phase 4).

Build from the project root (the folder containing app/ and models/):

    pyinstaller packaging/LocalRedactor.spec --noconfirm

Notes
-----
* Bundles `models/` (GLiNER2 weights, OCR onnx, tessdata) so the app runs fully
  offline. Place the model files there before building.
* RapidOCR/onnxruntime and gliner2/transformers ship data files and dynamic
  submodules; we collect them explicitly so the frozen app can find them.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

PROJECT_ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH injected by PyInstaller

datas = []
binaries = []
hiddenimports = []

# Bundle offline model assets if present.
models_dir = PROJECT_ROOT / "models"
if models_dir.exists():
    datas.append((str(models_dir), "models"))

# Collect data files + submodules for packages PyInstaller cannot fully trace.
for pkg in ("rapidocr_onnxruntime", "onnxruntime", "gliner2", "transformers", "tokenizers"):
    try:
        datas += collect_data_files(pkg)
        hiddenimports += collect_submodules(pkg)
    except Exception:
        # Optional dependency not installed in this build environment.
        pass

hiddenimports += ["fitz", "img2pdf", "PIL", "rapidfuzz", "regex", "pydantic"]


a = Analysis(
    [str(PROJECT_ROOT / "app" / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(PROJECT_ROOT / "packaging" / "pyi_rth_offline.py")],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalRedactor",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no console window
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="LocalRedactor",
)
