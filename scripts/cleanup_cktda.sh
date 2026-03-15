#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${AUTOCKT_CLEANUP_ROOT:-/tmp/ckt_da}"
INTERVAL_SEC="${AUTOCKT_CLEANUP_INTERVAL_SEC:-45}"
START_PCT="${AUTOCKT_CLEANUP_START_PCT:-80}"
STOP_PCT="${AUTOCKT_CLEANUP_STOP_PCT:-70}"
MIN_AGE_MIN="${AUTOCKT_CLEANUP_MIN_AGE_MIN:-3}"
MAX_DELETE_PER_ROUND="${AUTOCKT_CLEANUP_MAX_DELETE_PER_ROUND:-150}"

usage_pct() {
  df -P "${ROOT_DIR}" 2>/dev/null | awk 'NR==2 {gsub("%","",$5); print $5}'
}

delete_old_batch() {
  local deleted=0
  # Only delete sufficiently old design directories.
  # Excludes the root and only considers directories named "designs_*".
  while IFS= read -r d; do
    rm -rf -- "$d" || true
    deleted=$((deleted + 1))
    if [ "$deleted" -ge "$MAX_DELETE_PER_ROUND" ]; then
      break
    fi
  done < <(find "${ROOT_DIR}" -mindepth 2 -maxdepth 2 -type d -name "designs_*" -mmin "+${MIN_AGE_MIN}" | sort)
  echo "$deleted"
}

mkdir -p "${ROOT_DIR}"
echo "[cleanup] daemon started: root=${ROOT_DIR}, interval=${INTERVAL_SEC}s, start=${START_PCT}%, stop=${STOP_PCT}%"

while true; do
  pct="$(usage_pct || echo 0)"
  if [ "${pct}" -ge "${START_PCT}" ]; then
    echo "[cleanup] usage=${pct}% >= ${START_PCT}%, start cleanup cycle"
    while [ "${pct}" -ge "${STOP_PCT}" ]; do
      n="$(delete_old_batch)"
      pct="$(usage_pct || echo 0)"
      echo "[cleanup] deleted=${n}, usage_now=${pct}%"
      # Nothing left to delete, stop busy loop.
      if [ "${n}" -eq 0 ]; then
        break
      fi
      sleep 2
    done
  fi
  sleep "${INTERVAL_SEC}"
done

