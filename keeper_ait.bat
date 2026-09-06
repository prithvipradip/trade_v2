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
REM R16: size-cap keeper.log. The keeper only ever appended (>>) and nothing
REM else rotated this file -- 22.7 MB / ~448k lines and growing, inside a
REM 438 MB logs directory on a machine that must stay up for the bot. Pure
REM batch, no PowerShell and no cmd chaining operators: %%~zA is the size in
REM bytes.
REM Shift keeper.log -> .1 -> .2 above 20 MB; master.py's daily sweep caps the
REM backups. Runs before the ping so a rotation can never delay the check.
set KLSIZE=0
for %%A in ("logs\keeper.log") do set KLSIZE=%%~zA
if "%KLSIZE%"=="" set KLSIZE=0
if %KLSIZE% GTR 20971520 (
    if exist "logs\keeper.log.2" del "logs\keeper.log.2"
    if exist "logs\keeper.log.1" move /y "logs\keeper.log.1" "logs\keeper.log.2" >nul
    move /y "logs\keeper.log" "logs\keeper.log.1" >nul
    echo [keeper] %date% %time% rotated keeper.log at %KLSIZE% bytes >> logs\keeper.log
)

REM Is an orchestrator process alive?
REM A11 (deep-audit): wmic is deprecated/removed on newer Windows -- its
REM disappearance would have silently broken the keeper AND the dup-guard.
REM R16 #10: also match 'ait.main' -- a bot child orphaned by a master-only
REM death did NOT match 'run_orchestrator', so the keeper relaunched a full
REM second stack on top of it = TWO trading bots on the same account. If
REM either process exists, do not launch; BotManager refuses alongside an
REM orphan bot and alerts, so the operator resolves it instead of a blind
REM double-spawn.
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_orchestrator|ait\.main' }) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [keeper] %date% %time% orchestrator DOWN - relaunching >> logs\keeper.log
    start "" /min %PY% run_orchestrator.py
) else (
    echo [keeper] %date% %time% orchestrator alive >> logs\keeper.log
    REM R6: external dead-man ping. Create a check at healthchecks.io and put
    REM its ping URL (one line) in data\deadman_url.txt -- the service alerts
    REM when pings STOP: machine off, hard reboot at logon screen, keeper
    REM dead, or bot down.
    REM W6/bot-day-02: the ping used to fire whenever the PROCESS existed, so
    REM it attested the SUPERVISOR was alive -- not that the bot was working.
    REM A Telegram-dead + gateway-down outage produced a green external
    REM monitor for hours while positions sat unmanaged. Now gated on real
    REM liveness evidence (heartbeat within 900s = 30 missed beats, RTH only,
    REM post-open warmup) -- the SAME threshold master.py already uses for
    REM bot_hung_heartbeat_stale, so this adds no new alert surface.
    REM Not-alive sends /fail so the check alerts now instead of waiting out
    REM its grace period.
    if exist data\deadman_url.txt (
        %PY% -m ait.monitoring.ops_health liveness >nul 2>&1
        if errorlevel 1 (
            for /f "usebackq delims=" %%u in ("data\deadman_url.txt") do curl.exe -fsS -m 10 "%%u/fail" >nul 2>&1
            echo [keeper] %date% %time% deadman: bot NOT demonstrably alive - sent /fail >> logs\keeper.log
        ) else (
            for /f "usebackq delims=" %%u in ("data\deadman_url.txt") do curl.exe -fsS -m 10 "%%u" >nul 2>&1
        )
    )
)
REM R16: `timeout /t` needs an interactive console input handle. In some
REM console states (stdin redirected / detached) it fails INSTANTLY with
REM "ERROR: Input redirection is not supported", and this loop then spun at
REM ~1-2 iterations/sec: keeper.log gained ~144k lines on 08-04 and 6,254 in
REM the single hour 20:00-21:00 on 08-05 (normal is 40/hour). Once
REM data\deadman_url.txt is armed that same tight loop would curl
REM healthchecks.io ~once per second -- rate-limited into false "down" alerts
REM from the one channel that has to stay reliable.
REM Three independent delays, cheapest first, none needing console input.
REM `ping -n 91` waits ~1s between echoes = ~90s. No cmd chaining operators
REM anywhere (parser-safe): each fallback is guarded by `if errorlevel 1`.
ping -n 91 127.0.0.1 >nul 2>&1
if errorlevel 1 timeout /t 90 /nobreak >nul 2>&1
if errorlevel 1 powershell -NoProfile -Command "Start-Sleep -Seconds 90"
goto loop
