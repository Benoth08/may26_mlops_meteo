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
from pathlib import Path

logger = get_logger("make_dataset")
 
PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
MODELS_DIR = Path(SETTINGS["paths"]["models"])

DATASET_OUTPUT_PATH = (PROCESSED_DIR / SETTINGS["models"]["dataset"])
PREPROCESSOR_OUTPUT_PATH = (MODELS_DIR / SETTINGS["models"]["preprocessor"])


def main():
    
    # -----------------------------
    # 1. Lire les variables d'env
    # ----------------------------- 
    try:
        cfg = load_postgres_config()
    except ConfigError as e:
        # On ne logue jamais les valeurs des identifiants, seulement les
        # noms des variables manquantes (cf. config.py) : l'ancienne version
        # affichait POSTGRES_WTH_PASSWORD en clair dès qu'une AUTRE variable
        # (ex: POSTGRES_WTH_HOST) était absente.
        logger.error({"event": "config_error", "error": str(e)})
        sys.exit(1)
        
    connstring = cfg.sqlalchemy_uri
    
    logger.info({"event": "make_dataset_start", "host": cfg.host, "db": cfg.db})
    
    try:
        data = prepare_data(
            source="postgres",
            connection_uri=connstring,
            table_name=SETTINGS["postgres"]["table_raw"],
            save_report=True,
        )
    except Exception as e:
        logger.error({"event": "prepare_data_failed", "error": str(e)}, exc_info=True)
        sys.exit(1)

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
