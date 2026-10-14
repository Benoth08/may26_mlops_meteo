from pathlib import Path
import os
import yaml


def _resolve_params_file() -> Path:
    """Localise params.yml quel que soit le contexte d'ex?cution."""

    candidates = []

    # 1. Chemin explicite ?ventuel
    env_path = os.getenv("PARAMS_FILE")
    if env_path:
        candidates.append(Path(env_path))

    # 2. Ex?cution depuis la racine du d?p?t
    candidates.append(Path.cwd() / "params.yml")

    # 3. Images Docker o? params.yml est copi? ? c?t? du package parent
    candidates.append(
        Path(__file__).resolve().parent.parent / "params.yml"
    )

    # 4. Structure du d?p?t : <repo>/src/core/params.py -> <repo>/params.yml
    candidates.append(
        Path(__file__).resolve().parents[2] / "params.yml"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n - ".join(str(p) for p in candidates)

    raise FileNotFoundError(
        "params.yml introuvable. Chemins test?s :\n - " + checked
    )


PARAMS_FILE = _resolve_params_file()


def load_params():
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
