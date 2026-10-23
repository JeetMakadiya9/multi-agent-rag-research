@echo off
rem Starts the Research Studio backend and its dashboard on this computer (127.0.0.1 only).
rem Options are passed on, for example: start_studio.bat --studio-port 8766 --dashboard-port 8502
rem Stop both with Ctrl+C in this window.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [studio] No virtual environment found in "%~dp0.venv".
    echo [studio] Create it once:
    echo            py -3.12 -m venv .venv
    echo            .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

".venv\Scripts\python.exe" -m dashboard.launch %*
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo [studio] Stopped with exit code %EXITCODE%. See README.md, section "Troubleshooting".
    pause
)
exit /b %EXITCODE%
