@echo off
cd /d "%~dp0"
echo Starting OpenENVI...
python -m app.main %*
if errorlevel 1 (
    echo OpenENVI exited with error code %errorlevel%
    pause
)
