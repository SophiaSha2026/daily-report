@echo off
REM ===========================================================================
REM  Backup trigger for the auction workflow. Does NOT rely on GitHub cron.
REM
REM  WHY: GitHub Actions schedule is best-effort. Observed 2026-08-24: the
REM  premarket cron fired 97 minutes late. Observed 2026-08-25: premarket plus
REM  the first three auction entries never fired at all before 08:46 BJT.
REM
REM  SCHEDULE: US/Eastern Sun-Thu 19:30.
REM    19:30 EDT (summer) = 07:30 Beijing next day
REM    19:30 EST (winter) = 08:30 Beijing next day
REM  Both land before the 09:19:40 BJT T1 snapshot, so the local wall-clock
REM  time never needs adjusting for daylight saving.
REM
REM  Sun-Thu is deliberate. US Monday evening is Beijing Tuesday morning, so
REM  a Mon-Fri local schedule would MISS Beijing Monday and waste a run on
REM  Beijing Saturday.
REM
REM  Re-triggering is harmless: the workflow has a serial concurrency group
REM  plus an idempotency check. If today's snapshot is already committed the
REM  whole job is skipped, so no second email.
REM
REM  Uses the workflow FILE NAME (auction.yml), not its display name, because
REM  the display name is Chinese and .cmd files are read in the OEM codepage.
REM
REM  Log:    tools/trigger.log (gitignored)
REM
REM  Usage:  trigger_auction.cmd          fire the workflow
REM          trigger_auction.cmd check    verify gh auth and connectivity only
REM ===========================================================================
setlocal
set "REPO=SophiaSha2026/daily-report"
REM Log next to the script via %~dp0. Do NOT use %LOCALAPPDATA% here:
REM under Task Scheduler the write silently did not happen (gh itself ran
REM fine and the workflow was dispatched, but no log line appeared).
REM %~dp0 is expanded by cmd itself and never depends on the environment.
set "LOG=%~dp0trigger.log"
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format s"') do set "NOW=%%i"

if /i "%~1"=="check" (
  echo [%NOW%] check >> "%LOG%"
  gh auth status >> "%LOG%" 2>&1
  gh workflow list -R %REPO% >> "%LOG%" 2>&1
  echo [%NOW%] check exit=%ERRORLEVEL% >> "%LOG%"
  goto :eof
)

echo [%NOW%] trigger >> "%LOG%"
gh workflow run auction.yml --ref main -R %REPO% >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%NOW%] trigger exit=%RC% >> "%LOG%"
if not "%RC%"=="0" (
  echo [%NOW%] FAILED - GitHub cron entries from 07:40 BJT are the fallback >> "%LOG%"
)
