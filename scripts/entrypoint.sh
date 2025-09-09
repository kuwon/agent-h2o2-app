#!/usr/bin/env sh
# set -eu  # <- -e 빼서 조기종료 방지
set -u

CMD="${1:-serve}"

# --- NEW: DB URL 구성 & 대기 ---
# --- NEW: DB URL 구성 (A안: DB_* 조합 → DATABASE_URL 생성) ---
build_db_url() {
  if [ -z "${DATABASE_URL:-}" ] && [ -n "${DB_HOST:-}" ] && [ -n "${DB_USER:-}" ] && [ -n "${DB_PASS:-}" ]; then
    H="$DB_HOST"; P="${DB_PORT:-5432}"; D="${DB_DATABASE:-postgres}"
    export DATABASE_URL="postgresql+psycopg://${DB_USER}:${DB_PASS}@${H}:${P}/${D}?sslmode=require"
    echo "[entrypoint] DATABASE_URL constructed for ${H}:${P}/${D}"
  fi
}

# --- NEW: DB 대기 (DATABASE_URL 또는 DB_HOST가 있을 때만) ---
wait_for_db() {
  if [ -n "${DATABASE_URL:-}" ]; then
    python - <<'PY'
import os, time, socket, urllib.parse
url=os.environ["DATABASE_URL"]
# driver(+psycopg) 제거 후 파싱
parsed=urllib.parse.urlparse(url.replace("+psycopg",""))
host=parsed.hostname; port=parsed.port or 5432
for _ in range(60):
    try:
        with socket.create_connection((host, int(port)), timeout=2): pass
        print("[entrypoint] DB up"); raise SystemExit(0)
    except Exception: time.sleep(2)
raise SystemExit("[entrypoint] DB not reachable")
PY
  elif [ -n "${DB_HOST:-}" ]; then
    python - <<'PY'
import os, time, socket
host=os.environ["DB_HOST"]; port=int(os.environ.get("DB_PORT","5432"))
for _ in range(60):
    try:
        with socket.create_connection((host, int(port)), timeout=2): pass
        print("[entrypoint] DB up"); raise SystemExit(0)
    except Exception: time.sleep(2)
raise SystemExit("[entrypoint] DB not reachable")
PY
  else
    echo "[entrypoint] No DATABASE_URL/DB_HOST provided; skip DB wait."
    # 만약 DB 없으면 반드시 실패시키고 싶다면 다음 줄 주석 해제:
    # exit 1
  fi
}

start_api() {
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips "*" &
  API_PID=$!
}
start_ui() {
  streamlit run ui/pages/H2O2_pension_master.py --server.address 0.0.0.0 --server.port 8501 --server.headless true &
  UI_PID=$!
}
stop_all() {
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
  [ -n "${UI_PID:-}" ] && kill "$UI_PID" 2>/dev/null || true
}
trap stop_all TERM INT

case "$CMD" in
  serve)
    build_db_url
    wait_for_db || true         # DB 없다고 컨테이너 죽이지 않음
    start_api
    start_ui
    wait "$API_PID" || true     # 둘 중 하나 실패해도 컨테이너 유지
    wait "$UI_PID"  || true
    ;;
  api)
    build_db_url
    wait_for_db || true
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips "*"
    ;;
  ui)
    build_db_url
    streamlit run ui/pages/H2O2_pension_master.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
    ;;
  chill)
    tail -f /dev/null
    ;;
  *)
    exec "$@"
    ;;
esac
