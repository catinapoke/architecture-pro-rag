#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOCK_DIR="${LOG_DIR}/update_index.lock"
LOG_FILE="${LOG_DIR}/update_index.log"

mkdir -p "${LOG_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') | WARN | update is already running" >> "${LOG_FILE}"
  exit 0
fi

cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${PROJECT_ROOT}"
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | cron update started" >> "${LOG_FILE}"
"${PROJECT_ROOT}/.venv/bin/python" update_index.py >> "${LOG_FILE}" 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO | cron update finished" >> "${LOG_FILE}"

# Example crontab entry (daily at 06:00):
# 0 6 * * * /absolute/path/to/architecture-pro-rag/scripts/update_index_cron.sh
