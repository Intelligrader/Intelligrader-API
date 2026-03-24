from __future__ import annotations

import logging
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


# Constants
MAX_LOADED_MODELS = 3  # maximum number of GGUF files retained on disk at once
MODELS_DIR = Path("/app/model_storage")  # where GGUFs are stored on the volume
MODELS_YAML = Path(__file__).parent / "models.yaml"

logger = logging.getLogger(__name__)

# Ensure the storage directory always exists.
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _load_and_validate_models() -> tuple[dict[str, dict], str]:
    try:
        raw = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load models config from {MODELS_YAML}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise RuntimeError("Invalid models.yaml format: expected top-level 'models' list")

    models_by_id: dict[str, dict] = {}
    default_ids: list[str] = []

    for entry in raw["models"]:
        if not isinstance(entry, dict):
            raise RuntimeError("Invalid models.yaml format: each model entry must be a mapping")

        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeError("Invalid models.yaml format: each model must have a non-empty string 'id'")

        if model_id in models_by_id:
            raise RuntimeError(f"Duplicate model id in models.yaml: {model_id}")

        models_by_id[model_id] = entry

        if entry.get("default") is True:
            default_ids.append(model_id)

    if len(default_ids) != 1:
        raise RuntimeError(
            f"models.yaml must define exactly one default model, found {len(default_ids)}"
        )

    return models_by_id, default_ids[0]


_MODELS_BY_ID, _DEFAULT_MODEL_ID = _load_and_validate_models()


def is_supported(model_id: str) -> bool:
    return model_id in _MODELS_BY_ID


def get_default_model_id() -> str:
    return _DEFAULT_MODEL_ID


def get_model_config(model_id: str) -> dict:
    if not is_supported(model_id):
        raise ValueError(f"Unsupported model: {model_id}")

    return _MODELS_BY_ID[model_id]


def get_model_path(model_id: str) -> Path:
    if not is_supported(model_id):
        raise ValueError(f"Unsupported model: {model_id}")

    filename = _MODELS_BY_ID[model_id]["filename"]
    model_path = MODELS_DIR / filename

    if model_path.exists():
        return model_path

    _download_model(model_id)
    return model_path


def _download_model(model_id: str) -> None:
    if not is_supported(model_id):
        raise ValueError(f"Unsupported model: {model_id}")

    _evict_if_needed()

    model_cfg = _MODELS_BY_ID[model_id]
    repo_id = model_cfg["repo_id"]
    filename = model_cfg["filename"]

    logger.info("Downloading %s from %s...", model_id, repo_id)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=MODELS_DIR,
    )


def _evict_if_needed() -> None:
    gguf_files = [p for p in MODELS_DIR.glob("*.gguf") if p.is_file()]

    if len(gguf_files) >= MAX_LOADED_MODELS:
        oldest_file = min(gguf_files, key=lambda p: p.stat().st_mtime)
        logger.info(
            "Evicting %s to make room (limit=%s)", oldest_file.name, MAX_LOADED_MODELS
        )
        oldest_file.unlink(missing_ok=True)
