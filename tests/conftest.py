"""Charge l'environnement du projet avant de collecter les tests.

core.settings lit plusieurs variables d'environnement (WEATHER_HOST_*,
AIRFLOW_CONTAINER_*) directement via os.environ[...] à l'import du module ;
sans elles, `import core.settings` lève un KeyError avant même de pouvoir
collecter les tests.

De même, core.params.load_params() résout params.yml relativement à
l'emplacement physique de core/ (parent.parent), ce qui suppose une mise en
page de déploiement où params.yml est copié à côté de core/ (voir
src/dockerfiles/*/Dockerfile). En environnement de dev / CI, params.yml vit
à la racine du dépôt : on corrige le chemin avant le premier appel.
"""
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

import core.params  # noqa: E402

core.params.PARAMS_FILE = ROOT_DIR / "params.yml"
