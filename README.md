# Intelligrader API Service

FastAPI service for Intelligrader AI text generation using a local GGUF model via `llama-cpp-python`.

## Live URLs

- Production API: [https://peral.one](https://peral.one)
- API docs (Swagger): [https://peral.one/docs](https://peral.one/docs)

## Project Overview

This repository contains:

- A FastAPI app (`main.py`) with:
  - Health endpoint with model-readiness checks
  - Text generation endpoint powered by `llama_cpp.Llama`
- Docker packaging (`Dockerfile`)
- Production and staging Compose stacks:
  - `docker-compose.yml`
  - `docker-compose.staging.yml`
- NGINX reverse proxy configs for production and staging:
  - `nginx/conf.d/peral.one.conf`
  - `nginx/conf.d/staging.peral.one.conf`
- GitHub Actions deployment workflows in `.github/workflows/`

## API Endpoints

### `GET /`

Redirects to `/docs`.

### `GET /health`

Readiness-style health endpoint.

- Returns `200` when model is loaded:
  ```json
  {
    "status": "healthy",
    "model": "loaded"
  }
  ```
- Returns `503` when model is not loaded:
  ```json
  {
    "status": "unhealthy",
    "model": "not loaded",
    "error": "..."
  }
  ```

### `POST /generate`

Generate text from a prompt.

Request body:

```json
{
  "prompt": "Explain recursion in one sentence.",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Response:

```json
{
  "prompt": "Explain recursion in one sentence.",
  "generated_text": "...",
  "tokens_used": 42
}
```

If model is unavailable, returns `503`.

## Quick Start

Test API health:

```bash
curl -i https://peral.one/health
```

Test generation:

```bash
curl -s https://peral.one/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello from Intelligrader","max_tokens":32,"temperature":0.7,"top_p":0.9}'
```

## Local Development

### Requirements

- Python 3.11+
- Build tooling for `llama-cpp-python` (CMake, compiler toolchain)
- A valid GGUF model file at:
  - `models/SmolLM2-Rethink-360M.F32.gguf`

Python dependencies are listed in `requirements.txt`:

- `fastapi==0.104.1`
- `uvicorn[standard]==0.24.0`
- `llama-cpp-python==0.2.90`

### Install and run

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## Model File and Git LFS

The GGUF model is tracked with Git LFS. If you clone without pulling LFS artifacts, the model file will be a small text pointer, not real model bytes.

The app now detects this and reports an explicit startup error.

After clone, make sure model artifacts are present:

```bash
git lfs pull
```

## Docker

### Build and run API container

```bash
docker build -t intelligrader-api .
docker run --rm -p 8000:8000 intelligrader-api
```

The container starts:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Full production stack (API + NGINX)

```bash
docker compose -f docker-compose.yml up -d --build
```

### Full staging stack

```bash
docker compose -f docker-compose.staging.yml up -d --build
```

## Deployment and CI/CD

GitHub Actions workflows:

- `deploy.yml` deploys production on push to `main`
- `deploy-staging.yml` deploys staging on push to `staging`
- `status-check.yml` runs post-deploy checks for:
  - Endpoint reachability (`/health`, `/docs`)
  - Generation smoke test via `POST /generate`

## Troubleshooting

### `/health` returns `503`

Most common causes:

- Model file missing at `./models/SmolLM2-Rethink-360M.F32.gguf`
- Model file is still an LFS pointer (`git lfs pull` not run)
- `llama_cpp` failed to initialize due to incompatible environment

Check container logs:

```bash
docker logs <container_name>
```

### `/generate` returns `503`

Model is not loaded. Check `/health` response `error` field for root cause.

### `/generate` returns `500`

Runtime generation failure. Verify:

- Prompt/payload format
- Model integrity
- Available memory/CPU resources

## Security and Networking Notes

- NGINX is configured to enforce HTTPS and proxy traffic to FastAPI.
- Direct unmatched host requests are rejected in the provided NGINX configs.
- Health endpoint logging is disabled in proxy config to reduce noise.