#!/usr/bin/env python3
"""Étape 1 — Prépare les données via build_features (branche preprocessing).
Produit les splits train/test temporels + le préprocesseur NON entraîné."""

import sys
import json

from core.logger import get_logger
from core.settings import SETTINGS
from core.config import load_postgres_config, ConfigError, PostgresConfig
from build_features import prepare_data

import joblib
import pandas as pd
from pathlib import Path

logger = get_logger("make_dataset")

DATA_DIR = Path(SETTINGS["paths"]["data"])
PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
MODELS_DIR = Path(SETTINGS["paths"]["models"])

RAWDATASET_INPUT_PATH = (DATA_DIR / SETTINGS["models"]["raw_dataset"])
DATASET_OUTPUT_PATH = (PROCESSED_DIR / SETTINGS["models"]["dataset"])
PREPROCESSOR_OUTPUT_PATH = (MODELS_DIR / SETTINGS["models"]["preprocessor"])

# Code de sortie signalant à Airflow (DockerOperator.skip_on_exit_code) qu'il
# n'y a aucune donnée à traiter : la tâche et les tâches en aval doivent être
# marquées "skipped", pas "failed".
SKIP_EXIT_CODE = 99


def main():

    # -----------------------------
    # 1. Lire le fichier parquet
    # -----------------------------
    logger.info({"event": "make_dataset_start", "rawdataset": str(RAWDATASET_INPUT_PATH)})

    if not RAWDATASET_INPUT_PATH.exists():
        logger.error({"event": "Fichier non disponible", "path": str(RAWDATASET_INPUT_PATH)})
        sys.exit(1)

    if pd.read_parquet(RAWDATASET_INPUT_PATH).empty:
        logger.warning({
            "event": "make_dataset_skipped",
            "message": "Dataset brut vide, aucun traitement effectué.",
            "path": str(RAWDATASET_INPUT_PATH),
        })
        print("⚠️  make_dataset SKIPPED (0 ligne, aucun traitement)")
        sys.exit(SKIP_EXIT_CODE)

    try:
        data = prepare_data(
            source="parquet",
            data_path=str(RAWDATASET_INPUT_PATH),
            save_report=True,
        )
    except Exception as e:
        logger.error({"event": "prepare_data_failed", "error": str(e)}, exc_info=True)
        sys.exit(1)

    # -
    # 2. Enregistrement des données 
    # -
    DATASET_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {k: data[k] for k in ("X_train", "X_test", "y_train", "y_test")},
        DATASET_OUTPUT_PATH
    )
    joblib.dump(data["preprocessor"], PREPROCESSOR_OUTPUT_PATH)

    logger.info({
        "event": "make_dataset_done",
        "X_train_shape": tuple(data["X_train"].shape),
        "X_test_shape": tuple(data["X_test"].shape),
        "dataset_path": str(DATASET_OUTPUT_PATH),
        "preprocessor_path": str(PREPROCESSOR_OUTPUT_PATH),
    })
    
    print("✅ make_dataset OK")
    print("   X_train :", data["X_train"].shape, "| X_test :", data["X_test"].shape)


if __name__ == "__main__":
    main()
