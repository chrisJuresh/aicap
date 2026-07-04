@echo off
setlocal
set SCRIPT_DIR=%~dp0

if "%~1"=="" (
  echo Usage: run_captioner.bat "C:\Path\To\video.mp4"
  echo.
  echo Tip: You can drag a video file onto this .bat file.
  echo Edit settings.toml to change defaults.
  pause
  exit /b 1
)

"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%src\video_captioner.py" "%~1" ^
  --settings-file "%SCRIPT_DIR%settings.toml" ^
  --settings-profile single ^
  --open-output-browser

pause
