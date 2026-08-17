import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def resolve_model_path(config: dict, base_dir: Path) -> Path:
    return base_dir / config['robot']['model_path']
