#!/usr/bin/env sh
set -eu

CMD="${1:-serve}"

start_api() {
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips "*" &
  API_PID=$!
}

start_ui() {
  streamlit run ui/Home.py --server.address 0.0.0.0 --server.port 8501 --server.headless true &
  UI_PID=$!
}

stop_all() {
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
  [ -n "${UI_PID:-}" ] && kill "$UI_PID" 2>/dev/null || true
}

trap stop_all TERM INT

case "$CMD" in
  serve)
    start_api
    start_ui
    wait "$API_PID"
    wait "$UI_PID"
    ;;
  api)
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips "*"
    ;;
  ui)
    streamlit run ui/Home.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
    ;;
  chill)
    tail -f /dev/null
    ;;
  *)
    exec "$@"
    ;;
esac
