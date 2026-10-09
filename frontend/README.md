# React dashboard

This Vite/React app is the primary web interface for the LLM graph compiler.
It reads available models and optimization metrics from `api_server.py`,
renders original and optimized DAGs, and plots structural, runtime, memory,
stress-sweep, and XLA comparison metrics.

From the repository root, start both the local API and dashboard with
`run-server.bat`. The browser UI is at `http://localhost:5173`; API requests
are proxied from Vite to `http://127.0.0.1:8000`.

Run a production bundle with `npm --prefix frontend run build`.
