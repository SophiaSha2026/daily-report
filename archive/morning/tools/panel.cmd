@echo off
REM Desktop entry point for the console.
REM
REM Keep this file PURE ASCII: cmd.exe reads .cmd in the OEM code page, so
REM Chinese comments turn into mojibake and any echo of them prints garbage.
REM
REM Build paths from %~dp0 only. This file was once broken because the literal
REM path was written with an escape that turned \t in "\tools\" into a TAB,
REM producing "C:\home\daily-report<TAB>ools\panel.ps1" -- PowerShell could not
REM find the script, errored, and the window closed before anything was read.
REM
REM The window must survive a non-zero exit, otherwise the error is invisible.
title A-share pipeline console
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0panel.ps1"
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo.
  echo [!] Console exited with code %RC%
  echo     Common causes: gh not installed / gh not logged in / execution policy
  echo     Check with:    gh auth status
  echo.
  pause
)
