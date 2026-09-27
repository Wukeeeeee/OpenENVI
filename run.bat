@echo off
cd /d "%~dp0"
echo Starting OpenENVI...
python -m app.main data\benchmark_hyper.hdr
if errorlevel 1 (
    echo OpenENVI exited with error code %errorlevel%
    pause
)
