#!/usr/bin/env bash
# Stop the Ray head node, Grafana, Prometheus and MLflow started by start_monitoring.sh.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RAY="$REPO/venv/bin/ray"
MON_DIR="$REPO/.cache/ray-monitoring"

"$RAY" stop

if [[ -f "$MON_DIR/grafana.pid" ]]; then
  kill "$(cat "$MON_DIR/grafana.pid")" 2>/dev/null && echo "Stopped Grafana"
  rm -f "$MON_DIR/grafana.pid"
fi

"$RAY" metrics shutdown-prometheus

# MLflow runs a server process plus web workers; stop them all by their command line.
pkill -f "mlflow server --host 127.0.0.1 --port 5050" && echo "Stopped MLflow"
pkill -f "mlflow.server.*5050|uvicorn.*mlflow.server" 2>/dev/null || true
