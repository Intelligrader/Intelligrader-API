#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${1:-}"
PROJECT_NAME="${2:-}"
TAIL_LINES="${3:-200}"

if [ -z "$COMPOSE_FILE" ] || [ -z "$PROJECT_NAME" ]; then
  echo "Usage: $0 <compose-file> <project-name> [tail-lines]"
  echo "Example: $0 docker-compose.staging.yml intelligrader-staging 300"
  exit 2
fi

echo "=== Docker Compose ps (${PROJECT_NAME}) ==="
docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" ps || true

echo "=== Docker Compose logs (${PROJECT_NAME}) tail=${TAIL_LINES} ==="
docker compose -p "$PROJECT_NAME" -f "$COMPOSE_FILE" logs --no-color --tail "$TAIL_LINES" || true