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
REM
REM  DISPATCH WINDOW (added 2026-09-04): only dispatch between 07:30 and
REM  09:16 Beijing time. The workflow spins inside the job until 09:27:30 and
REM  its timeout is 175 minutes; a dispatch at 06:57 (logon trigger fired when
REM  the laptop woke up) would have been killed seconds before sending. The
REM  15-minute retry loop keeps knocking, so the first attempt inside the
REM  window still goes out. After 09:16 a fresh job cannot reach T1 anyway.
REM ===========================================================================
setlocal
set "REPO=SophiaSha2026/daily-report"
set "LOG=%~dp0trigger.log"
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().AddHours(8).ToString('yyyy-MM-dd')"') do set "BJDATE=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "(Get-Date).ToUniversalTime().AddHours(8).ToString('HHmm')"') do set "BJHM=%%i"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format s"') do set "NOW=%%i"
set "BJMONTH=%BJDATE:~0,7%"

REM Prefix with 1 so cmd compares decimally (a leading 0 would mean octal).
if 1%BJHM% LSS 10730 (
  echo [%NOW%] auction: %BJDATE% %BJHM% BJT before window 07:30, skip >> "%LOG%"
  goto :eof
)
if 1%BJHM% GTR 10916 (
  echo [%NOW%] auction: %BJDATE% %BJHM% BJT after window 09:16, skip >> "%LOG%"
  goto :eof
)

REM Already produced today? Then there is nothing to trigger.
gh api "repos/%REPO%/contents/data/%BJMONTH%/auction_%BJDATE%.parquet" --silent >nul 2>&1
if "%ERRORLEVEL%"=="0" (
  echo [%NOW%] auction: %BJDATE% already done, skip >> "%LOG%"
  goto :eof
)

echo [%NOW%] auction trigger for %BJDATE% >> "%LOG%"
gh workflow run auction.yml --ref main -R %REPO% >> "%LOG%" 2>&1
echo [%NOW%] auction exit=%ERRORLEVEL% >> "%LOG%"
