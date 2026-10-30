#!/usr/bin/env bash
# Stop the Ray head node, Grafana and Prometheus started by start_monitoring.sh.
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
