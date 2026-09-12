@echo off
REM ===========================================================================
REM  Desktop entry point for the console (GUI). Replaces panel.cmd, which
REM  launched the PowerShell TUI. The TUI still works and is kept as a
REM  fallback for when a browser is not available.
REM
REM  Keep this file PURE ASCII: cmd.exe reads .cmd in the OEM code page, so
REM  Chinese comments turn into mojibake and any echo of them prints garbage.
REM
REM  Build paths from %~dp0 only. panel.cmd was once broken by a literal path
REM  whose "\tools\" became a TAB, producing a path PowerShell could not find.
REM
REM  This window stays open while the console runs -- closing it stops the
REM  server (and any flow started BY HAND from the browser). Scheduled tasks
REM  are separate processes and are NOT affected.
REM ===========================================================================
title A-share pipeline console
chcp 65001 >nul

set "PY=C:\Users\xueji\AppData\Local\Programs\Python\Python311\python.exe"
if not exist "%PY%" set "PY=python"

REM Unbuffered, otherwise the URL line sits in the buffer and the window
REM looks empty until the first request comes in.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"

cd /d "%~dp0.."
"%PY%" src\gui\__main__.py %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [!] Console exited with code %RC%
  echo     Port already in use?  try:  tools\gui.cmd --port 8766
  echo.
  pause
)
