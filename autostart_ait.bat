@echo off
REM AIT v2 boot auto-start launcher (invoked by the "AIT Trading Bot"
REM scheduled task at logon). Silent, with a duplicate-supervisor guard so a
REM manual run + the scheduled run don't both spawn an orchestrator.

cd /d C:\Users\prith\Documents\Git\agent_trade\trade_v2
set PYTHONIOENCODING=utf-8

REM Guard: bail if a run_orchestrator.py is already running.
wmic process where "name='python.exe' and commandline like '%%run_orchestrator.py%%'" get processid 2>nul | findstr /r "[0-9]" >nul
if %errorlevel%==0 (
    echo [autostart] orchestrator already running, skipping. >> logs\autostart.log
    exit /b 0
)

echo [autostart] %date% %time% launching orchestrator >> logs\autostart.log
start "" /min python run_orchestrator.py
