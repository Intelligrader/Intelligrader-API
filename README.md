# Intelligrader API Service

Simple FastAPI service for Intelligrader

## Quick Start
Test our API's integrity with the following command:
```bash
curl -I https://peral.one/health
```

## Our API is hosted at: [https://peral.one](https://peral.one)
## You can see the docs at: [https://peral.one/docs](https://peral.one/docs)

## Deployment Notes
This repository uses Docker Compose managed container names to avoid name conflicts.

Use explicit project names when deploying:

Production:
- docker compose -f docker-compose.yml --project-name intelligrader-prod up -d --build

Staging:
- docker compose -f docker-compose.staging.yml --project-name intelligrader-staging up -d --build

If old manually named containers still exist on the server, remove them once:
- docker rm -f intelligrader-api intelligrader-api-staging

## Endpoints
Please Check Back Soon