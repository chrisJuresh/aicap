@echo off
setlocal
set SCRIPT_DIR=%~dp0

"%SCRIPT_DIR%.venv\Scripts\python.exe" -m unittest discover -s "%SCRIPT_DIR%tests" -p "test_*.py"

pause
