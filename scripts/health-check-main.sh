#!/usr/bin/env bash
set -euo pipefail

URL="https://peral.one/health"
MAX_ATTEMPTS=15
SLEEP_SECONDS=10

echo "Checking main health endpoint: ${URL}"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  response_file="$(mktemp)"

  http_code=$(curl -sS -o "$response_file" -w "%{http_code}" "$URL" || true)
  response_body=$(cat "$response_file")
  rm -f "$response_file"

  if [ "$http_code" = "200" ] && echo "$response_body" | grep -q '"status":"healthy"'; then
    echo "Main is healthy on attempt ${attempt}."
    echo "$response_body"
    exit 0
  fi

  echo "Attempt ${attempt}/${MAX_ATTEMPTS} failed (HTTP ${http_code})."
  echo "Response: ${response_body}"

  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    sleep "$SLEEP_SECONDS"
  fi
done

echo "Main health check failed after ${MAX_ATTEMPTS} attempts."
exit 1