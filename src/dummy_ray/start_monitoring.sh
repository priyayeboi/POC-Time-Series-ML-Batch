#!/usr/bin/env bash
# Start a local Ray head node with Prometheus metrics and Grafana dashboards.
#
#   Ray dashboard: http://127.0.0.1:8265  (Metrics tab embeds the Grafana panels)
#   Prometheus:    http://localhost:9090
#   Grafana:       http://localhost:3000  (dashboards in the "Ray" folder)
#
# Run from anywhere: src/dummy_ray/start_monitoring.sh
# Training scripts connect to this cluster with ray.init().
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RAY="$REPO/venv/bin/ray"
MON_DIR="$REPO/.cache/ray-monitoring"
GRAFANA_HOME="$(brew --prefix grafana)/share/grafana"
mkdir -p "$MON_DIR"

# Ray's dashboard reads these at startup to embed Grafana panels and query Prometheus.
export RAY_GRAFANA_HOST=http://localhost:3000
export RAY_PROMETHEUS_HOST=http://localhost:9090

# 1. Ray head node. Restart it so the variables above take effect.
"$RAY" stop >/dev/null 2>&1 || true
"$RAY" start --head --metrics-export-port=8080 --dashboard-host=127.0.0.1

# 2. Prometheus, using the scrape config Ray generates. Reuse one that is already running.
if curl -sf http://localhost:9090/-/ready >/dev/null; then
  echo "Prometheus already running on :9090"
else
  (cd "$MON_DIR" && "$RAY" metrics launch-prometheus)
fi

# 3. Grafana, restarted so it loads this Ray session's dashboard provisioning.
if [[ -f "$MON_DIR/grafana.pid" ]]; then
  kill "$(cat "$MON_DIR/grafana.pid")" 2>/dev/null || true
  sleep 1
fi
nohup grafana server \
  --homepath "$GRAFANA_HOME" \
  --config /tmp/ray/session_latest/metrics/grafana/grafana.ini \
  cfg:default.paths.data="$MON_DIR/grafana-data" \
  cfg:default.paths.logs="$MON_DIR/grafana-logs" \
  web >"$MON_DIR/grafana.log" 2>&1 &
echo $! >"$MON_DIR/grafana.pid"

for _ in $(seq 1 30); do
  curl -sf http://localhost:3000/api/health >/dev/null && break
  sleep 1
done
curl -sf http://localhost:3000/api/health >/dev/null || { echo "Grafana did not start; see $MON_DIR/grafana.log"; exit 1; }

echo
echo "Ray dashboard: http://127.0.0.1:8265"
echo "Prometheus:    http://localhost:9090"
echo "Grafana:       http://localhost:3000"
