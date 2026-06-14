@echo off
REM Build a standalone LocalRedactor.exe with PyInstaller (Phase 4).
REM Produces dist\LocalRedactor\LocalRedactor.exe (one-folder build).
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3.11 -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

REM Make sure model assets exist (offline). Warn but continue if missing.
if not exist "models\gliner2-pii-v1" (
    echo [warning] models\gliner2-pii-v1 not found. Bundle it for offline GLiNER2.
)

python -m PyInstaller packaging\LocalRedactor.spec --noconfirm --clean

echo.
echo Build complete: dist\LocalRedactor\LocalRedactor.exe
endlocal
