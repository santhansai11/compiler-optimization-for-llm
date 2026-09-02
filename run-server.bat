@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PATH=C:\Program Files\Graphviz\bin;%PATH%"
set "VENV=.venv312"
set "PY=%VENV%\Scripts\python.exe"
set "ST=%VENV%\Scripts\streamlit.exe"

if not exist "%PY%" (
  echo Creating Python 3.12 virtualenv...
  py -3.12 -m venv "%VENV%"
  if errorlevel 1 (
    echo Failed to create venv. Install Python 3.12 and the "py" launcher.
    exit /b 1
  )
)

if not exist "%ST%" (
  echo Installing dependencies from requirements.txt...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo pip install failed.
    exit /b 1
  )
)

echo Starting LLM Compiler Optimizer at http://localhost:8501
"%ST%" run app.py --server.port 8501
