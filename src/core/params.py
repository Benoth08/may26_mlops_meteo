from pathlib import Path
import os
import yaml


def _resolve_params_file() -> Path:
    """Localise params.yml quel que soit le contexte d'ex?cution."""

    candidates = []

    # 1. Chemin explicite éventuel
    env_path = os.getenv("PARAMS_FILE")
    if env_path:
        candidates.append(Path(env_path))

    # 2. Exécution depuis la racine du dépôt
    candidates.append(Path.cwd() / "params.yml")

    # 3. Images Docker où params.yml est copié à coté du package parent
    candidates.append(
        Path(__file__).resolve().parent.parent / "params.yml"
    )

    # 4. Structure du dépôt : <repo>/src/core/params.py -> <repo>/params.yml
    candidates.append(
        Path(__file__).resolve().parents[2] / "params.yml"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n - ".join(str(p) for p in candidates)

    raise FileNotFoundError(
        "params.yml introuvable. Chemins testés :\n - " + checked
    )


PARAMS_FILE = _resolve_params_file()


def load_params():
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
