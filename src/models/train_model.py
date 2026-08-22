"""Étape 3 — Entraîne le pipeline complet (préprocesseur + LightGBM) avec les
meilleurs hyperparamètres, sur tout le train."""
import os
import sys
 
from core.logger import get_logger
from core.settings import SETTINGS
from core.metadata import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET,
)

# Doit être posé AVANT l'import de numpy/lightgbm/sklearn.
for _env_key, _env_val in SETTINGS["threads"].items():
    os.environ.setdefault(_env_key, str(_env_val))

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


from pathlib import Path

from datetime import datetime, timezone

from platform import python_version
from sklearn import __version__ as sklearn_version
from joblib import __version__ as joblib_version
from numpy import __version__ as numpy_version
from pandas import __version__ as pandas_version

import joblib


logger = get_logger("train_model")

PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
MODELS_DIR = Path(SETTINGS["paths"]["models"])

DATASET_PATH = (PROCESSED_DIR / SETTINGS["models"]["dataset"])
PREPROCESSOR_PATH = (MODELS_DIR / SETTINGS["models"]["preprocessor"])
BEST_PARAMS_PATH = (MODELS_DIR / SETTINGS["models"]["best_params"])
MODEL_OUTPUT_PATH = (MODELS_DIR / SETTINGS["models"]["model"])

def build_model_metadata(data):
    """
    Métadonnées nécessaires au serving du modèle.
    """

    location_column = SETTINGS["location"]["column_norm"]

    if location_column not in data["X_train"].columns:
        raise ValueError(
            f"La colonne {location_column} est absente du dataset train."
        )

    known_locations = sorted(
        data["X_train"][location_column]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),

        "python_version": python_version(),
        "sklearn_version": sklearn_version,
        "numpy_version": numpy_version,
        "pandas_version": pandas_version,
        "joblib_version": joblib_version,
        
        "model_version": "1.0.0",
        "target": SETTINGS["target"]["column_norm"],
        "features": {
            "numeric": NUMERIC_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
        },
        "location": {
            "column": location_column,
            "known_values": known_locations,
            "count": len(known_locations),
        }, 
        "training": {
            "train_rows": len(data["X_train"]),
            "test_rows": len(data["X_test"]),
        },
    }
    

def main():
    
    logger.info({"event": "loading_model_train", "dataset_path": str(DATASET_PATH)})
    
    try:
        data = joblib.load(DATASET_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        best_params = joblib.load(BEST_PARAMS_PATH)
    except FileNotFoundError as e:
        logger.error({"event": "artifact_not_found", "error": str(e)}, exc_info=True)
        sys.exit(1)

    lgbm_params = {k.replace("model__", ""): v for k, v in best_params.items()}

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            class_weight="balanced", random_state=SETTINGS["seed"], n_jobs=-1, verbosity=-1, **lgbm_params)),
    ])
    
    try:
        pipe.fit(data["X_train"], data["y_train"])
    except Exception as e:
        logger.error({"event": "fit_failed", "error": str(e)}, exc_info=True)
        sys.exit(1)   
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_model_metadata(data)
    artifact = {
        "pipeline": pipe,
        "metadata": metadata
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)

    logger.info({"event": "model_trained", "output_path": str(MODEL_OUTPUT_PATH)})
    
    print("✅ train_model OK -> ", str(MODEL_OUTPUT_PATH))
    print("   Hyperparamètres :", lgbm_params)


if __name__ == "__main__":
    main()
