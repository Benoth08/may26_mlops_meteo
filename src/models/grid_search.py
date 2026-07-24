"""Étape 2 — GridSearch (TimeSeriesSplit). Pipeline = préprocesseur + LightGBM,
re-fitté sur le train de chaque pli -> aucune fuite."""

import os
import sys

import joblib
from pathlib import Path

from core.settings import SETTINGS
from core.logger import get_logger

# Important : LGBMClassifier(n_jobs=1) + GridSearchCV(n_jobs=-1) ci-dessous
# parallélise au niveau des plis/combinaisons, pas au niveau de chaque
# modèle. Pour éviter l'oversubscription CPU (chaque process qui relance
# des threads BLAS/OMP par-dessus), on force explicitement les libs
# numériques à 1 thread. Doit être posé AVANT l'import de numpy/lightgbm.
for _env_key, _env_val in SETTINGS["threads"].items():
    os.environ.setdefault(_env_key, str(_env_val))

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier



logger = get_logger("grid_search")

PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
MODELS_DIR = Path(SETTINGS["paths"]["models"])

DATASET_PATH = (PROCESSED_DIR / SETTINGS["models"]["dataset"])
PREPROCESSOR_PATH = (MODELS_DIR / SETTINGS["models"]["preprocessor"])
BEST_PARAMS_PATH = (MODELS_DIR / SETTINGS["models"]["best_params"])
MODEL_OUTPUT_PATH = (MODELS_DIR / SETTINGS["models"]["model"])


def main():
    
    logger.info({"event": "loading_grid_search", "dataset_path": str(DATASET_PATH)})
    
    try :
        data = joblib.load(DATASET_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except FileNotFoundError as e:
        logger.error({"event": "artifact_not_found", "error": str(e)}, exc_info=True)
        sys.exit(1)
        
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            class_weight="balanced", random_state=SETTINGS["seed"], n_jobs=1, verbosity=-1)),
    ])

    grid = {"model__n_estimators": [100, 200], "model__num_leaves": [31, 63]}
    cv = TimeSeriesSplit(n_splits=3)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    
    try:
        search.fit(data["X_train"], data["y_train"])
    except Exception as e:
        logger.error({"event": "grid_search_failed", "error": str(e)}, exc_info=True)
        sys.exit(1)

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_params_, BEST_PARAMS_PATH)

    logger.info({
        "event": "grid_search_ended",
        "best_params": search.best_params_,
        "best_score_f1_cv": float(search.best_score_),
    })
    
    print("✅ grid_search OK")
    print("   Meilleurs paramètres :", search.best_params_)
    print("   F1 (CV) :", round(float(search.best_score_), 4))


if __name__ == "__main__":
    main()
