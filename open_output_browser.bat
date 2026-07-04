@echo off
setlocal
set SCRIPT_DIR=%~dp0

set "TARGET=%~1"
if "%TARGET%"=="" (
  if exist "%SCRIPT_DIR%output_batch\index.html" (
    set "TARGET=%SCRIPT_DIR%output_batch\index.html"
  ) else if exist "%SCRIPT_DIR%output\index.html" (
    set "TARGET=%SCRIPT_DIR%output\index.html"
  )
) else (
  if exist "%TARGET%\index.html" (
    set "TARGET=%TARGET%\index.html"
  )
)

if "%TARGET%"=="" (
  echo No output browser found yet.
  echo.
  echo Run run_batch_captioner.bat or run_batch_captioner_10.bat first.
  echo Expected: output_batch\index.html or output\index.html
  pause
  exit /b 1
)

if not exist "%TARGET%" (
  echo Output browser not found:
  echo %TARGET%
  pause
  exit /b 1
)

start "" "%TARGET%"
