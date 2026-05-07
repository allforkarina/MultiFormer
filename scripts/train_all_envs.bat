@echo off
setlocal
cd /d "%~dp0\.."

for %%E in (E01 E02 E03 E04) do (
    echo === Training %%E ===
    python train.py --config configs\%%E.yaml
    if errorlevel 1 exit /b 1
)
