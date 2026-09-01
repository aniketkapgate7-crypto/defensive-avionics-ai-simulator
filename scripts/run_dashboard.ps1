# Multi-Modal Defensive Avionics AI Simulator — Launcher
# Safe classroom simulation launcher script

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   MULTI-MODAL DEFENSIVE AVIONICS AI SIMULATOR" -ForegroundColor Cyan
Write-Host "   Offline Synthetic Academic Simulation" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Set-Location $ProjectRoot

# Locate Python in virtual environment or system
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvStreamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"

if (Test-Path $VenvPython) {
    Write-Host "[OK] Using virtual environment: .venv" -ForegroundColor Green
    $PythonExe = $VenvPython
} else {
    Write-Host "[INFO] Virtual environment not found at .venv. Trying system python..." -ForegroundColor Yellow
    $PythonExe = "python"
}

# Verify Streamlit presence
$StreamlitInstalled = & $PythonExe -c "import streamlit; print('OK')" 2>$null
if ($StreamlitInstalled -ne "OK") {
    Write-Host "[!] Installing UI dependencies (Streamlit & Pygame)..." -ForegroundColor Yellow
    & $PythonExe -m pip install -e ".[ui]"
}

Write-Host "`nLaunching Engineering HUD Dashboard..." -ForegroundColor Cyan
Write-Host "`nSYNTHETIC CLASSROOM SIMULATION - NO REAL-WORLD TARGETING`n" -ForegroundColor Yellow

& $PythonExe -m streamlit run app/dashboard.py --server.headless=false
