$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
  git rev-parse --is-inside-work-tree | Out-Null
  git config core.hooksPath .githooks
  Write-Host "Git hooks installed. The pre-commit and pre-push guards will block local/private or sensitive files."
} finally {
  Pop-Location
}
