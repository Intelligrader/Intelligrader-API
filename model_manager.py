import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import yaml
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger("model_manager")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_repo: str
    filename: str
    active: bool

    @property
    def local_path(self) -> Path:
        return Path("models") / self.filename


def _read_registry(registry_path: Path) -> List[ModelSpec]:
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}

    raw_models = payload.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("models.yaml must contain a top-level 'models' list")

    specs: List[ModelSpec] = []
    for entry in raw_models:
        if not isinstance(entry, dict):
            raise ValueError("Each model entry must be a mapping")

        specs.append(
            ModelSpec(
                name=str(entry["name"]),
                hf_repo=str(entry["hf_repo"]),
                filename=str(entry["filename"]),
                active=bool(entry.get("active", False)),
            )
        )

    return specs





def _download_model_if_missing(spec: ModelSpec, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / spec.filename

    if target_path.exists():
        LOGGER.info("Model file already exists, skipping download: %s", target_path)
        return target_path

    LOGGER.info("Downloading model %s from %s", spec.filename, spec.hf_repo)
    downloaded_path = hf_hub_download(
        repo_id=spec.hf_repo,
        filename=spec.filename,
        local_dir=str(models_dir),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded_path)


def ensure_models(registry_path: str = "models.yaml", models_dir: str = "models") -> List[ModelSpec]:
    registry = _read_registry(Path(registry_path))
    active_models = [spec for spec in registry if spec.active]

    if not active_models:
        LOGGER.warning("No active models found in %s", registry_path)
        return []

    model_dir_path = Path(models_dir)

    for spec in active_models:
        _download_model_if_missing(spec, model_dir_path)

    return active_models


def get_active_local_model_path(models: Iterable[ModelSpec], models_dir: str = "models") -> Path:
    active_list = list(models)
    if not active_list:
        raise ValueError("No active models available")

    return Path(models_dir) / active_list[0].filename


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_models()
