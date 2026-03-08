param(
    [string]$PythonExe = "python",
    [string]$MountLibrary = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"

if (-not (Test-Path $VenvPython)) {
    & $PythonExe -m venv (Join-Path $RepoRoot ".venv")
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
& $VenvPython (Join-Path $RepoRoot "constellation.py") paths reset

if ($MountLibrary) {
    & $VenvPython (Join-Path $RepoRoot "constellation.py") paths mount-library $MountLibrary
}

Write-Host "Constellation bootstrap complete."
Write-Host "Launcher: $VenvPython $((Join-Path $RepoRoot 'constellation.py')) realtime --tray"
