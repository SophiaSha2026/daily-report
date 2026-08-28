@echo off
REM ===========================================================================
REM  Backup trigger for the call-auction workflow. Retries across the window.
REM
REM  WHY REPEATED RETRIES INSTEAD OF ONE SHOT:
REM  A single daily trigger only fires if the laptop happens to be awake at
REM  that exact instant. It was not, on 2026-08-27, and both the auction and
REM  the pattern email were missed. WakeToRun could not help either: this
REM  machine had wake timers disabled on battery and set to "important only"
REM  on AC, and a plain scheduled task never counts as important.
REM  Wake timers are now enabled, but a powered-off machine still cannot be
REM  woken. So the task now retries across the whole usable window and also
REM  fires on logon. The machine only has to be awake at SOME point.
REM
REM  Re-triggering is free: this script asks GitHub whether today's data file
REM  already exists and exits without dispatching if it does. Even if it did
REM  dispatch, the workflow has a serial concurrency group and its own
REM  idempotency check.
REM
REM  Log: tools/trigger.log (gitignored)
REM ===========================================================================
setlocal
set "REPO=SophiaSha2026/daily-report"
set "LOG=%~dp0trigger.log"
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().AddHours(8).ToString('yyyy-MM-dd')"') do set "BJDATE=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format s"') do set "NOW=%%i"
set "BJMONTH=%BJDATE:~0,7%"

REM Already produced today? Then there is nothing to trigger.
gh api "repos/%REPO%/contents/data/%BJMONTH%/auction_%BJDATE%.parquet" --silent >nul 2>&1
if "%ERRORLEVEL%"=="0" (
  echo [%NOW%] auction: %BJDATE% already done, skip >> "%LOG%"
  goto :eof
)

echo [%NOW%] auction trigger for %BJDATE% >> "%LOG%"
gh workflow run auction.yml --ref main -R %REPO% >> "%LOG%" 2>&1
echo [%NOW%] auction exit=%ERRORLEVEL% >> "%LOG%"
