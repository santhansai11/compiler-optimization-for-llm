@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Project Python environment was not found at %PY%.
  echo Create it and install requirements.txt before starting the site.
  exit /b 1
)
if not exist "frontend\node_modules\vite\bin\vite.js" (
  echo React dependencies are missing. Installing them with npm...
  npm --prefix frontend install
  if errorlevel 1 exit /b 1
)

echo Starting compiler API on http://127.0.0.1:8000
powershell -NoProfile -Command "$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if (-not $listener) { Start-Process -FilePath (Resolve-Path '%PY%') -ArgumentList 'api_server.py' -WorkingDirectory '%CD%' -WindowStyle Hidden; for ($attempt = 0; $attempt -lt 20; $attempt++) { if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { break }; Start-Sleep -Milliseconds 500 } }"
if errorlevel 1 (
  echo Could not start the compiler API.
  exit /b 1
)

echo Starting React dashboard at http://localhost:5173
echo Keep this window open while using the dashboard.
cd frontend
npm run dev -- --host 127.0.0.1
