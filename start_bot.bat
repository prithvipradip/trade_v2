@echo off
REM AIT v2 Master Orchestrator Launcher
REM Starts the full system: bot + scheduler + monitoring

cd /d C:\Users\prith\Documents\Git\agent_trade\trade_v2
set PYTHONIOENCODING=utf-8

echo ============================================
echo   AIT v2 Master Orchestrator
echo   Bot + ML Retrain + Backtest + Reports
echo ============================================
echo.

python run_orchestrator.py
