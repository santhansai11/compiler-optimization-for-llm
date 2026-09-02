@echo off
setlocal
cd /d "%~dp0"

set "PATH=C:\Program Files\Graphviz\bin;%PATH%"

if exist ".venv312\Scripts\streamlit.exe" (
  ".venv312\Scripts\streamlit.exe" run app.py --server.port 8501
  goto :eof
)

if exist ".venv\Scripts\streamlit.exe" (
  ".venv\Scripts\streamlit.exe" run app.py --server.port 8501
  goto :eof
)

echo No virtualenv found. Create one with:
echo   py -3.12 -m venv .venv312
echo   .venv312\Scripts\python.exe -m pip install streamlit networkx graphviz numpy
exit /b 1
