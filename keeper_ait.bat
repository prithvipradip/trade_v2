@echo off
REM ============================================================
REM  AIT Bot Keeper
REM  Relaunches run_orchestrator.py whenever it dies. Runs as a
REM  cmd.exe process (NOT python.exe), so it survives whatever is
REM  killing python on this machine on a ~30-60 min cycle. This is
REM  a band-aid over that machine-level issue, not a fix for it.
REM
REM  Checks every 90s. Logs to logs\keeper.log.
REM ============================================================
title AIT Bot Keeper
cd /d C:\Users\prith\Documents\Git\agent_trade\trade_v2
set PYTHONIOENCODING=utf-8
set PY="C:\Users\prith\AppData\Local\Programs\Python\Python313\python.exe"

:loop
REM Is an orchestrator process alive?
REM A11 (deep-audit): wmic is deprecated/removed on newer Windows -- its
REM disappearance would have silently broken the keeper AND the dup-guard.
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_orchestrator' }) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [keeper] %date% %time% orchestrator DOWN - relaunching >> logs\keeper.log
    start "" /min %PY% run_orchestrator.py
) else (
    echo [keeper] %date% %time% orchestrator alive >> logs\keeper.log
    REM R6: external dead-man ping. Create a check at healthchecks.io and put
    REM its ping URL (one line) in data\deadman_url.txt -- the service alerts
    REM when pings STOP: machine off, hard reboot at logon screen, keeper
    REM dead, or bot down. Pinged ONLY while the orchestrator is alive.
    if exist data\deadman_url.txt (
        for /f "usebackq delims=" %%u in ("data\deadman_url.txt") do curl.exe -fsS -m 10 "%%u" >nul 2>&1
    )
)
timeout /t 90 /nobreak >nul
goto loop
