#!/usr/bin/env bash
set -euo pipefail

# Clean up stale Ray state from previous container runs.
pkill -9 -f ray >/dev/null 2>&1 || true
pkill -9 -f gcs_server >/dev/null 2>&1 || true
pkill -9 -f raylet >/dev/null 2>&1 || true
rm -rf /tmp/ray || true

if [ "${AUTOCKT_CLEANUP_ENABLE:-1}" = "1" ]; then
  /usr/local/bin/cleanup_cktda.sh >/tmp/cleanup_cktda.log 2>&1 &
fi

exec "$@"
