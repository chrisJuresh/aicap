@echo off
setlocal
set SCRIPT_DIR=%~dp0

if "%~1"=="" (
  echo Usage: run_batch_captioner_10.bat "C:\Path\To\FolderOfVideos"
  echo.
  echo Tip: You can drag a folder onto this .bat file.
  echo This processes videos in groups of 10 so finished outputs appear sooner.
  pause
  exit /b 1
)

"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%src\video_captioner.py" "%~1" ^
  --settings-file "%SCRIPT_DIR%settings.toml" ^
  --settings-profile batch ^
  --batch-chunk-size 10 ^
  --open-output-browser

pause
