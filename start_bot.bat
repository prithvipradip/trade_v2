@echo off
REM AIT v2 Master Orchestrator Launcher
REM Starts the full system: bot + scheduler + monitoring

cd /d C:\Users\prith\Documents\Git\agent_trade\trade_v2
set PYTHONIOENCODING=utf-8

REM R5 audit F11: the AIT-Bot-Start scheduled task (07:30 daily) runs this file
REM with NO duplicate guard — it launched second orchestrators on top of the
REM keeper-owned one (confirmed 06-25, 06-26, 07-01). Same guard as autostart.
REM R16 #10: also match 'ait.main' — an orphaned bot child didn't match
REM 'run_orchestrator', so this guard passed and a second full stack (and
REM second trading bot) launched on top of it.
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_orchestrator|ait\.main' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo Orchestrator already running - not starting a duplicate.
    exit /b 0
)

echo ============================================
echo   AIT v2 Master Orchestrator
echo   Bot + ML Retrain + Backtest + Reports
echo ============================================
echo.

python run_orchestrator.py
