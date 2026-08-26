@echo off
REM ===========================================================================
REM  Backup trigger for the pattern-scan workflow (pullback.yml).
REM  Same idea as trigger_auction.cmd, different schedule. Kept as a separate
REM  file on purpose: the auction trigger is verified working and runs against
REM  a hard 40-minute window, so it is not worth refactoring to share code.
REM
REM  SCHEDULE: US/Eastern Mon-Fri 07:00.
REM    07:00 EDT (summer) = 19:00 Beijing SAME day
REM    07:00 EST (winter) = 20:00 Beijing SAME day
REM  Both sit inside the scan window (run_at 17:00, hard deadline 22:00 BJT),
REM  so the local wall-clock time never needs adjusting for daylight saving.
REM
REM  Mon-Fri here, NOT Sun-Thu. Beijing is 12-13 hours ahead, and 07:00 local
REM  plus 12-13 hours stays inside the same calendar day, so US weekdays map
REM  one-to-one onto Beijing weekdays. The auction trigger fires in the evening
REM  and therefore DOES cross midnight, which is why that one is Sun-Thu.
REM
REM  Re-triggering is harmless: serial concurrency group plus an idempotency
REM  check. If today's scan is already committed the whole job is skipped.
REM  Triggering too early is also harmless: the job refuses to spin-wait more
REM  than 45 minutes and exits clean with a warning.
REM
REM  Log:    tools/trigger.log (gitignored, shared with the auction trigger)
REM
REM  Usage:  trigger_pullback.cmd          fire the workflow
REM          trigger_pullback.cmd check    verify gh auth and connectivity only
REM ===========================================================================
setlocal
set "REPO=SophiaSha2026/daily-report"
REM %~dp0 is expanded by cmd itself and never depends on the environment.
REM %LOCALAPPDATA% silently failed to write under Task Scheduler. See
REM trigger_auction.cmd for the full note.
set "LOG=%~dp0trigger.log"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format s"') do set "NOW=%%i"

if /i "%~1"=="check" (
  echo [%NOW%] pullback check >> "%LOG%"
  gh workflow list -R %REPO% >> "%LOG%" 2>&1
  echo [%NOW%] pullback check exit=%ERRORLEVEL% >> "%LOG%"
  goto :eof
)

echo [%NOW%] pullback trigger >> "%LOG%"
gh workflow run pullback.yml --ref main -R %REPO% >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%NOW%] pullback trigger exit=%RC% >> "%LOG%"
if not "%RC%"=="0" (
  echo [%NOW%] FAILED - GitHub cron 17:00/17:30/18:20 BJT is the fallback >> "%LOG%"
)
