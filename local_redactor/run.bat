@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Enforce offline operation for the ML stack (Hard Invariant: no network calls).
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

python -m app.main %*

endlocal
