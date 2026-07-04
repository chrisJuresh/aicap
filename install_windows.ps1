[CmdletBinding()]
param(
  [string]$VisionModel = "huihui_ai/qwen2.5-vl-abliterated:latest",
  [string]$TextModel = "richardyoung/qwen3-14b-abliterated:Q4_K_M",
  [switch]$SkipModelPull
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Have-Command($Name) {
  return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Refresh-Path {
  $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
  $user = [Environment]::GetEnvironmentVariable("Path", "User")
  $env:Path = "$machine;$user"
}

function Install-WithWinget($Id, $FriendlyName) {
  if (-not (Have-Command "winget")) {
    Write-Warning "winget is not available. Install $FriendlyName manually, then re-run this script."
    return
  }
  Write-Host "Installing $FriendlyName via winget..." -ForegroundColor Cyan
  winget install --id $Id -e --accept-package-agreements --accept-source-agreements
  Refresh-Path
}

Write-Host "== Local Video Captioner Windows setup ==" -ForegroundColor Green

if (-not (Have-Command "python")) {
  Install-WithWinget "Python.Python.3.11" "Python 3.11"
}
Refresh-Path
if (-not (Have-Command "python")) {
  throw "Python was not found. Install Python 3.11+, then re-run this script."
}

if (-not (Have-Command "ffmpeg")) {
  Install-WithWinget "Gyan.FFmpeg" "FFmpeg"
}
Refresh-Path
if (-not (Have-Command "ffmpeg")) {
  Write-Warning "FFmpeg still was not found on PATH. You may need to open a new PowerShell window after install."
}

if (-not (Have-Command "ollama")) {
  Install-WithWinget "Ollama.Ollama" "Ollama"
}
Refresh-Path
if (-not (Have-Command "ollama")) {
  throw "Ollama was not found. Install Ollama from ollama.com/download, then re-run this script."
}

if (-not (Test-Path ".\.venv")) {
  Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
  python -m venv .venv
}

if ((Test-Path ".\settings.toml.example") -and -not (Test-Path ".\settings.toml")) {
  Write-Host "Creating local settings.toml from settings.toml.example..." -ForegroundColor Cyan
  Copy-Item ".\settings.toml.example" ".\settings.toml"
}

if ((Test-Path ".\prompts.toml.example") -and -not (Test-Path ".\prompts.toml")) {
  Write-Host "Creating local prompts.toml from prompts.toml.example..." -ForegroundColor Cyan
  Copy-Item ".\prompts.toml.example" ".\prompts.toml"
}

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Checking Ollama server..." -ForegroundColor Cyan
$ollamaUp = $false
try {
  Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 | Out-Null
  $ollamaUp = $true
} catch {
  Write-Host "Starting Ollama server..." -ForegroundColor Cyan
  Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized
  Start-Sleep -Seconds 6
  try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 10 | Out-Null
    $ollamaUp = $true
  } catch {
    Write-Warning "Could not confirm Ollama is running. Open the Ollama app or run 'ollama serve' in another terminal."
  }
}

if (-not $SkipModelPull) {
  Write-Host "Pulling vision model: $VisionModel" -ForegroundColor Cyan
  & ollama pull $VisionModel
  Write-Host "Pulling text model: $TextModel" -ForegroundColor Cyan
  & ollama pull $TextModel
} else {
  Write-Host "Skipping model pulls." -ForegroundColor Yellow
}

Write-Host "\nSetup complete." -ForegroundColor Green
Write-Host "Test Ollama: .\.venv\Scripts\python.exe .\src\test_ollama.py"
Write-Host "Run captioner: .\run_captioner.bat \"C:\Path\To\video.mp4\""
