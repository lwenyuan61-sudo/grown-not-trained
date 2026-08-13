@echo off
rem One-click reproduction (Windows). Double-click = quick demo; "run_all.bat full" = everything.
cd /d "%~dp0"
python run_all.py %*
pause
