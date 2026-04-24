# PharmaLink — Start All Services
# Run from the project root: .\start_all.ps1

$root      = $PSScriptRoot
$python    = "$root\.venv\Scripts\python.exe"
$streamlit = "$root\.venv\Scripts\streamlit.exe"

Write-Host "=== PharmaLink_GCN Platform ===" -ForegroundColor Cyan

# 1) AI_ENGINE FastAPI (port 8000)
Write-Host "`n[1/3] Starting AI_ENGINE (FastAPI, port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$root\AI_ENGINE'; & '$python' api.py`""

Start-Sleep -Seconds 3

# 2) BACKEND NestJS (port 3000)
Write-Host "[2/3] Starting BACKEND (NestJS, port 3000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$root\BACKEND'; npm run start:dev`""

Start-Sleep -Seconds 5

# 3) FRONTEND Streamlit (port 8501)
Write-Host "[3/3] Starting FRONTEND (Streamlit, port 8501)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit -Command `"cd '$root\FRONTEND'; & '$streamlit' run app.py`""

Write-Host "`nAll services launched!" -ForegroundColor Green
Write-Host "  FastAPI   -> http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  NestJS    -> http://localhost:3000/api"  -ForegroundColor Cyan
Write-Host "  Streamlit -> http://localhost:8501"      -ForegroundColor Cyan
