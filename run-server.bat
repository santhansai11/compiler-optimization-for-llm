@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title GraphForge

rem One file: create the environment if needed, then start the API and the dashboard.
set "ROOT=%~dp0"
set "PY=%ROOT%.venv312\Scripts\python.exe"
if not exist "%PY%" set "PY="

if defined PY (
  "%PY%" -c "import sys" >nul 2>&1
  if errorlevel 1 set "PY="
)

if not defined PY (
  echo Creating the Python 3.12 environment...
  where py >nul 2>&1
  if errorlevel 1 (
    echo Python launcher "py" was not found. Install Python 3.12, then run this file again.
    pause
    exit /b 1
  )
  py -3.12 -m venv "%ROOT%.venv312"
  if errorlevel 1 (
    echo Could not create .venv312.
    pause
    exit /b 1
  )
  set "PY=%ROOT%.venv312\Scripts\python.exe"
)

"%PY%" -c "import networkx, numpy, openpyxl, torch, jax, scipy, matplotlib, graphviz" >nul 2>&1
if errorlevel 1 (
  echo Installing Python packages. The first run can take several minutes...
  "%PY%" -m pip install -r "%ROOT%requirements.txt"
  if errorlevel 1 (
    echo Python package install failed.
    pause
    exit /b 1
  )
)

where npm >nul 2>&1
if errorlevel 1 (
  echo Node.js was not found. Install Node.js, then run this file again.
  pause
  exit /b 1
)

if not exist "%ROOT%frontend\node_modules\vite\bin\vite.js" (
  echo Installing the dashboard packages...
  call npm --prefix "%ROOT%frontend" install
  if errorlevel 1 (
    echo Dashboard package install failed.
    pause
    exit /b 1
  )
)

powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 8000); $c.Close(); exit 0 } catch { exit 1 }"
if errorlevel 1 (
  echo Starting the compiler API at http://127.0.0.1:8000
  start "GraphForge API" /D "%ROOT%" "%PY%" -u api_server.py
  powershell -NoProfile -Command "for ($i = 0; $i -lt 40; $i++) { try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 8000); $c.Close(); exit 0 } catch { Start-Sleep -Milliseconds 500 } }; exit 1"
  if errorlevel 1 (
    echo The API did not start. Check the GraphForge API window for the error.
    pause
    exit /b 1
  )
) else (
  echo Compiler API is already running at http://127.0.0.1:8000
)

powershell -NoProfile -Command "try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', 5173); $c.Close(); exit 0 } catch { exit 1 }"
if not errorlevel 1 (
  echo Dashboard is already running at http://127.0.0.1:5173
  start "" http://127.0.0.1:5173/
  echo You can close this window.
  pause
  exit /b 0
)

echo Starting the dashboard at http://127.0.0.1:5173
echo Leave this window open. Closing it stops the dashboard.
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5173/"
call npm --prefix "%ROOT%frontend" run dev -- --host 127.0.0.1
if errorlevel 1 (
  echo The dashboard stopped.
  pause
  exit /b 1
)
