#!/usr/bin/env bash
# One-command launcher (macOS/Linux): API on :8000, React GUI on :5173.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
[ -d .venv ] || { $PY -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
[ -d frontend/node_modules ] || npm --prefix frontend install
.venv/bin/python api_server.py &
API=$!
trap 'kill $API 2>/dev/null' EXIT
npm --prefix frontend run dev -- --host 127.0.0.1
