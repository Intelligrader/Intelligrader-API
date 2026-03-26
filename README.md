# Intelligrader API Service

FastAPI service for Intelligrader

## Quick Start
Test our API's integrity with the following command:
```bash
curl -I https://peral.one/health
```

## Our API is hosted at: [https://peral.one](https://peral.one)
## You can see the docs at: [https://peral.one/docs](https://peral.one/docs)

## Endpoints
Please Check Back Soon

## Request Queueing
- Generation requests are processed by a single worker queue to prevent concurrent model access races.
- Queue depth is exposed by `/health` and `/models` in `queue_depth`.
- The queue size limit is configurable with `MAX_QUEUE_SIZE` (default: `100`).
- When the queue is full, `/generate` returns HTTP `429`.