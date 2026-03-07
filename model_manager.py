import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

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


def _list_registered_ollama_models() -> Set[str]:
    process = subprocess.run(
        ["ollama", "list"],
        check=True,
        capture_output=True,
        text=True,
    )

    registered: Set[str] = set()
    for line in process.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("NAME"):
            continue

        model_name = stripped.split()[0]
        registered.add(model_name)
        if ":" in model_name:
            registered.add(model_name.split(":", 1)[0])

    return registered


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


def _register_with_ollama_if_missing(spec: ModelSpec, local_path: Path, registered_models: Set[str]) -> None:
    if spec.name in registered_models:
        LOGGER.info("Ollama model already registered, skipping create: %s", spec.name)
        return

    # Ollama expects a Modelfile with a FROM directive pointing at the GGUF file.
    modelfile_content = f"FROM {local_path.resolve().as_posix()}\n"
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as modelfile:
        modelfile.write(modelfile_content)
        modelfile_path = Path(modelfile.name)

    try:
        LOGGER.info("Registering model with Ollama: %s", spec.name)
        subprocess.run(
            ["ollama", "create", spec.name, "-f", str(modelfile_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        registered_models.add(spec.name)
    finally:
        modelfile_path.unlink(missing_ok=True)


def ensure_models(registry_path: str = "models.yaml", models_dir: str = "models") -> List[ModelSpec]:
    registry = _read_registry(Path(registry_path))
    active_models = [spec for spec in registry if spec.active]

    if not active_models:
        LOGGER.warning("No active models found in %s", registry_path)
        return []

    registered_models = _list_registered_ollama_models()
    model_dir_path = Path(models_dir)

    for spec in active_models:
        local_path = _download_model_if_missing(spec, model_dir_path)
        _register_with_ollama_if_missing(spec, local_path, registered_models)

    return active_models


def get_active_local_model_path(models: Iterable[ModelSpec], models_dir: str = "models") -> Path:
    active_list = list(models)
    if not active_list:
        raise ValueError("No active models available")

    return Path(models_dir) / active_list[0].filename


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_models()
