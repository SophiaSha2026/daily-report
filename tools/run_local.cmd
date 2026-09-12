@echo off
REM ===========================================================================
REM  Local flow runner. Called by Task Scheduler; replaces the old
REM  trigger_auction.cmd / trigger_pullback.cmd, which only DISPATCHED the
REM  cloud workflow. Since 2026-09-12 every pipeline runs on this machine and
REM  the repo is version control only, so there is nothing to dispatch.
REM
REM  Usage:  run_local.cmd morning | evening | learn
REM
REM  Keep this file PURE ASCII. cmd.exe reads .cmd in the OEM code page, so
REM  Chinese comments become mojibake and any echo of them prints garbage.
REM
REM  WHY --if-needed:
REM  The task retries every 15 minutes across the whole window because the
REM  laptop may be asleep at any given instant (2026-08-27: it was, and that
REM  day's mail never went out). Retrying means the entry point MUST be
REM  idempotent, otherwise a finished flow gets re-run -- and re-mailed --
REM  every 15 minutes. local_run.py --if-needed checks today's run_meta and
REM  exits 0 when the line is already done.
REM
REM  WHY AN ABSOLUTE PYTHON PATH:
REM  This box has three pythons on PATH: 3.11 (the real one), 3.14, and the
REM  WindowsApps stub that opens the Microsoft Store instead of running code.
REM  A scheduled task does not necessarily inherit the interactive PATH
REM  order, so resolve 3.11 explicitly and only fall back to PATH.
REM
REM  Log: tools/local_flow.log (gitignored)
REM ===========================================================================
setlocal
set "ROOT=%~dp0.."
set "LOG=%~dp0local_flow.log"
set "FLOW=%~1"
if "%FLOW%"=="" set "FLOW=morning"

set "PY=C:\Users\xueji\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PY%" set "PY=python"

REM Python defaults to the ANSI code page when stdout is a pipe; the log
REM would fill with mojibake and UnicodeEncodeError tracebacks.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format s"') do set "NOW=%%i"
echo [%NOW%] === run_local %FLOW% === >> "%LOG%"

cd /d "%ROOT%"
"%PY%" src\local_run.py --flow %FLOW% --if-needed >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"

for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format s"') do set "NOW=%%i"
echo [%NOW%] === run_local %FLOW% exit=%RC% === >> "%LOG%"
exit /b %RC%
