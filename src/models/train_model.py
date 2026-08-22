#!/usr/bin/env python3

"""
===============================================================================
Projet : mai26_continu_mlops / WeatherAUS
-------------------------------------------------------------------------------

Étape 3 — Entraînement du modèle final.

Responsabilités :
    - Charger le dataset prétraité
    - Charger le préprocesseur
    - Charger les meilleurs hyperparamètres issus du GridSearch
    - Construire le pipeline préprocesseur + LightGBM
    - Entraîner le modèle sur l'ensemble d'entraînement
    - Construire les métadonnées du modèle
    - Sauvegarder le modèle candidat dans models/candidate_model.joblib
      (seul promote_model.py le copie vers models/model.joblib, le
      fichier réellement servi, si la promotion est acceptée)

===============================================================================
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from platform import python_version
 
from core.logger import get_logger
from core.settings import SETTINGS
from core.metadata import (
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET,
)

# =============================================================================
# Imports ML
# =============================================================================

# Doit être posé AVANT l'import de numpy/lightgbm/sklearn.
for _env_key, _env_val in SETTINGS["threads"].items():
    os.environ.setdefault(_env_key, str(_env_val))

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

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
# Fichier candidat : promote_model.py décide s'il devient models/model.joblib.
MODEL_OUTPUT_PATH = (MODELS_DIR / SETTINGS["models"]["candidate_model"])

# Code de sortie signalant à Airflow (DockerOperator.skip_on_exit_code) qu'il
# n'y a aucune donnée à traiter : la tâche et les tâches en aval doivent être
# marquées "skipped", pas "failed".
SKIP_EXIT_CODE = 99

# Threads LightGBM pour ce fit unique. Doit rester aligné sur le "cpus=2" du
# DockerOperator train_model (weather_models_dag.py) : contrairement à
# grid_search.py où GridSearchCV(n_jobs=-1) parallélise déjà au niveau des
# plis/combinaisons (d'où LGBMClassifier(n_jobs=1) là-bas pour ne pas
# sur-souscrire), il n'y a ici qu'un seul fit, sans parallélisme au-dessus.
# n_jobs=-1 laisserait LightGBM lancer autant de threads que de coeurs
# détectés sur l'hôte (os.cpu_count() n'est pas conscient du quota cgroup
# Docker), au-delà du quota réellement alloué au conteneur.
LGBM_N_JOBS = 2

# =============================================================================
# CONTEXTE DU TRAINING
# =============================================================================
TRAINING_REASON = os.getenv("TRAINING_REASON", "manual",)
TRAINING_NEW_ROWS = int(os.getenv("TRAINING_NEW_ROWS", "0"))
TRAINING_ROW_COUNT = int(os.getenv("TRAINING_ROW_COUNT", "0"))


# =============================================================================
# METADATA
# =============================================================================
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
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        # ------------------------------------------------------------------
        # Versions techniques
        # ------------------------------------------------------------------
        "python_version": python_version(),
        "sklearn_version": sklearn_version,
        "numpy_version": numpy_version,
        "pandas_version": pandas_version,
        "joblib_version": joblib_version,

        # ------------------------------------------------------------------
        # Version du modèle
        # ------------------------------------------------------------------
        "model_version": "1.0.0",

        # ------------------------------------------------------------------
        # Target / features
        # ------------------------------------------------------------------
        "target": SETTINGS["target"]["column_norm"],
        "features": {
            "numeric": NUMERIC_COLUMNS,
            "categorical": CATEGORICAL_COLUMNS,
            "all": FEATURE_COLUMNS,
        },

        # ------------------------------------------------------------------
        # Localisation
        # ------------------------------------------------------------------
        "location": {
            "column": location_column,
            "known_values": known_locations,
            "count": len(known_locations),
        },

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        "training": {
            "train_rows": len(data["X_train"]),
            "test_rows": len(data["X_test"]),
            "training_reason": TRAINING_REASON,
            "new_rows": TRAINING_NEW_ROWS,
            "current_row_count": TRAINING_ROW_COUNT,
            "triggered_at": datetime.now(timezone.utc).isoformat()
        }
    }
    

# =============================================================================
# MAIN
# =============================================================================
def main():
    
    logger.info({
        "event": "loading_model_train",
        "dataset_path": str(DATASET_PATH),
        "training_reason": TRAINING_REASON,
        "new_rows": TRAINING_NEW_ROWS,
        "training_row_count": TRAINING_ROW_COUNT
    })
    
    # =========================================================================
    # 1. Chargement des artefacts
    # =========================================================================
    try:
        data = joblib.load(DATASET_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        best_params = joblib.load(BEST_PARAMS_PATH)
    except FileNotFoundError as e:
        # Artefact absent => grid_search (ou le preprocessing en amont) a été
        # "skipped" faute de données. Pas de données, pas d'entraînement.
        logger.warning({"event": "train_model_skipped", "reason": str(e)})
        print(f"⚠️  train_model SKIPPED (artefact manquant) : {e}")
        sys.exit(SKIP_EXIT_CODE)

    # =========================================================================
    # 2. Validation des données
    # =========================================================================
    required_keys = {
        "X_train",
        "y_train",
        "X_test",
        "y_test",
    }

    missing_keys = (
        required_keys
        - set(data.keys())
    )

    if missing_keys:
        logger.error(
            {
                "event": "invalid_dataset_artifact",
                "missing_keys": sorted(
                    missing_keys
                ),
            }
        )
        sys.exit(1)

    logger.info(
        {
            "event": "dataset_loaded",
            "train_rows": len(data["X_train"]),
            "test_rows": len(data["X_test"]),
            "features": len(data["X_train"].columns)
        }
    )

    # =========================================================================
    # 3. Préparation des hyperparamètres
    # =========================================================================
    if not isinstance(best_params, dict):
        logger.error(
            {
                "event": "invalid_best_params",
                "type": str(type(best_params))
            }
        )
        sys.exit(1)

    # Le GridSearch peut retourner des paramètres du type :
    #
    # model__n_estimators
    # model__learning_rate
    #
    # Le modèle LightGBM attend :
    #
    # n_estimators
    # learning_rate
    lgbm_params = {k.replace("model__", ""): v for k, v in best_params.items()}

    logger.info(
        {
            "event": "best_hyperparameters",
            "parameters": lgbm_params,
        }
    )

    # =========================================================================
    # 4. Construction du pipeline
    # =========================================================================
    try:
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", LGBMClassifier(
                class_weight="balanced", random_state=SETTINGS["seed"], n_jobs=LGBM_N_JOBS, verbosity=-1, **lgbm_params)),
        ])
    except Exception as error:
        logger.error(
            {
                "event": "pipeline_creation_failed",
                "error": str(error),
            },
            exc_info=True,
        )
        sys.exit(1)

    # =========================================================================
    # 5. Entraînement
    # =========================================================================
    logger.info(
        {
            "event": "model_training_started",
            "algorithm": "lightgbm",
            "train_rows": len(data["X_train"]),
            "training_reason": TRAINING_REASON
        }
    )
   
    try:
        pipe.fit(data["X_train"], data["y_train"])
    except Exception as e:
        logger.error({"event": "fit_failed", "error": str(e)}, exc_info=True)
        sys.exit(1)   
    
    # =========================================================================
    # 6. Création des métadonnées
    # =========================================================================
    metadata = build_model_metadata(data)
    # Ajouter les hyperparamètres réellement utilisés.
    metadata["model"] = {
        "algorithm": "lightgbm",
        "parameters": lgbm_params,
    }
    
    # =========================================================================
    # 7. Construction de l'artefact final
    # =========================================================================
    artifact = {
        "pipeline": pipe,
        "metadata": metadata
    }
    
    # =========================================================================
    # 8. Sauvegarde
    # =========================================================================
    try:
        MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, MODEL_OUTPUT_PATH)
        logger.info({"event": "model_trained", "output_path": str(MODEL_OUTPUT_PATH)})
    
    except Exception as error:
        logger.error(
            {
                "event": "model_save_failed",
                "output_path": str(MODEL_OUTPUT_PATH),
                "error": str(error),
            },
            exc_info=True,
        )
        sys.exit(1)
    
    # =========================================================================
    # 9. Fin
    # =========================================================================
    logger.info(
        {
            "event": "model_trained",
            "output_path": str(MODEL_OUTPUT_PATH),
            "algorithm": "lightgbm",
            "training_reason": TRAINING_REASON,
            "new_rows": TRAINING_NEW_ROWS,
            "training_row_count": TRAINING_ROW_COUNT,
            "train_rows": len(data["X_train"]),
            "test_rows": len(data["X_test"])
        }
    )

    print("✅ train_model OK -> ", str(MODEL_OUTPUT_PATH))
    print("   Algorithme : LightGBM")
    print("   Hyperparamètres :", lgbm_params)
    print("   Training reason :", TRAINING_REASON)
    print("   Nouvelles lignes :", TRAINING_NEW_ROWS)
    print("   Nombre total de lignes :", TRAINING_ROW_COUNT)
    

# =============================================================================
# ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    main()
