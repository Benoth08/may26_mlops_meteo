#!/usr/bin/env python3
"""
===============================================================================
    WeatherAUS — Prétraitement final (imputation, mise à l'échelle, encodage)
    ---------------------------------------------------------------------------
    Sujet :
        Applique le préprocesseur scikit-learn (imputation + scaling +
        encodage) sur les jeux X_train / X_test déjà splittés par
        make_dataset.py, sauvegarde les tableaux transformés et le
        préprocesseur fitté, puis (optionnellement) stocke les features
        obtenues dans PostgreSQL.

===============================================================================
"""

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

from core.logger import get_logger
from core.settings import SETTINGS
from core.config import load_postgres_config

from build_features import build_preprocessor, identify_feature_types

import joblib
import pandas as pd

logger = get_logger("normalize_data")

# ============================================================
# Constantes issues de SETTINGS (aucune valeur en dur)
# ============================================================

PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
MODELS_DIR = Path(SETTINGS["paths"]["models"])
DATASET_NAME = SETTINGS["models"]["dataset"]
PREPROCESSOR_NAME = SETTINGS["models"]["preprocessor"]
RANDOM_SEED = SETTINGS["seed"]
TABLE_FEATURES_ML = SETTINGS["postgres"]["table_features"]
TARGET = SETTINGS["target"]["column_norm"]

DATASET_PATH = PROCESSED_DIR / DATASET_NAME

X_TRAIN_SCALED_PATH = PROCESSED_DIR / "X_train_scaled.csv"
X_TEST_SCALED_PATH = PROCESSED_DIR / "X_test_scaled.csv"
PREPROCESSOR_PATH = MODELS_DIR / PREPROCESSOR_NAME


# ============================================================
# Traitement ML
# ============================================================

def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge X_train / X_test produits par make_dataset.py."""

    logger.info({
        "event": "loading_dataset_train_test",
        "dataset_path": str(DATASET_PATH)
    })

    # Test existence fichier
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    try:
        dataset = joblib.load(DATASET_PATH)

    except Exception as e:
        logger.error({
            "event": "dataset_loading_failed",
            "error": str(e)
        })
        raise RuntimeError(
            f"Impossible de charger le dataset : {DATASET_PATH}"
        ) from e


    # Test structure du fichier joblib
    if not isinstance(dataset, dict):
        raise TypeError(
            "Le dataset joblib doit contenir un dictionnaire."
        )


    required_keys = {"X_train", "X_test"}

    missing_keys = required_keys - dataset.keys()

    if missing_keys:
        raise KeyError(
            f"Clés manquantes dans le dataset : {missing_keys}"
        )


    X_train = dataset["X_train"]
    X_test = dataset["X_test"]


    # Test types pandas
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(
            f"X_train doit être un DataFrame, reçu : {type(X_train)}"
        )

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"X_test doit être un DataFrame, reçu : {type(X_test)}"
        )


    # Test dataset non vide
    if X_train.empty:
        raise ValueError("X_train est vide.")

    if X_test.empty:
        raise ValueError("X_test est vide.")


    logger.info({
        "event": "dataset_loaded",
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "features_count": len(X_train.columns)
    })


    return X_train, X_test



def load_target() -> tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Charge y_train / y_test depuis le dataset joblib.

    Les targets restent optionnelles pour le preprocessing.
    """

    logger.info({
        "event": "loading_targets",
        "dataset_path": str(DATASET_PATH)
    })


    if not DATASET_PATH.exists():
        logger.warning({
            "event": "dataset_not_found",
            "dataset_path": str(DATASET_PATH)
        })
        return None, None


    try:
        dataset = joblib.load(DATASET_PATH)

    except Exception as e:
        logger.error({
            "event": "dataset_loading_failed",
            "error": str(e)
        })
        return None, None


    if not isinstance(dataset, dict):
        logger.warning({
            "event": "invalid_dataset_format"
        })
        return None, None


    required_keys = {"y_train", "y_test"}

    missing_keys = required_keys - dataset.keys()

    if missing_keys:
        logger.warning({
            "event": "target_keys_missing",
            "missing_keys": list(missing_keys)
        })
        return None, None


    y_train = dataset["y_train"]
    y_test = dataset["y_test"]


    # Conversion DataFrame 1 colonne -> Series
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze("columns")

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze("columns")


    # Vérification finale
    if not isinstance(y_train, pd.Series):
        logger.warning({
            "event": "invalid_y_train_type",
            "type": str(type(y_train))
        })
        return None, None


    if not isinstance(y_test, pd.Series):
        logger.warning({
            "event": "invalid_y_test_type",
            "type": str(type(y_test))
        })
        return None, None


    logger.info({
        "event": "targets_loaded",
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape
    })


    return y_train, y_test
    
def fit_transform_preprocessor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame
):
    """
    Identifie les features numériques/catégorielles du schéma (metadata.py),
    construit le préprocesseur mutualisé (build_features.build_preprocessor)
    et l'applique : fit sur X_train uniquement, transform sur X_test
    (anti fuite de données).
    """
    numeric_features, categorical_features = identify_feature_types(X_train)

    # Garde-fou : ne garder que les colonnes réellement présentes dans le DF
    numeric_features = [c for c in numeric_features if c in X_train.columns]
    categorical_features = [c for c in categorical_features if c in X_train.columns]

    logger.info({
        "event": "building_preprocessor",
        "numeric_features_count": len(numeric_features),
        "categorical_features_count": len(categorical_features),
    })

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    logger.info({
        "event": "before_preprocessing",
        "X_train_shape": X_train.shape,
        "y_train_shape": None if y_train is None else y_train.shape,
        "y_train_type": str(type(y_train))
    })

    X_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)
    colonnes = preprocessor.get_feature_names_out()

    return preprocessor, X_train_scaled, X_test_scaled, colonnes


def save_scaled_datasets(
    X_train_scaled,
    X_test_scaled,
    colonnes,
    train_output: Path = X_TRAIN_SCALED_PATH,
    test_output: Path = X_TEST_SCALED_PATH,
) -> None:
    """Sauvegarde les jeux transformés au format CSV."""
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train_scaled, columns=colonnes).to_csv(train_output, index=False)
    pd.DataFrame(X_test_scaled, columns=colonnes).to_csv(test_output, index=False)

    logger.info({
        "event": "scaled_datasets_saved",
        "train_output": str(train_output),
        "test_output": str(test_output),
    })


def save_preprocessor(
    preprocessor,
    output_path: Path = PREPROCESSOR_PATH,
) -> None:
    """Sauvegarde le préprocesseur fitté pour l'inférence (Airflow / API)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)

    logger.info({"event": "preprocessor_saved", "path": str(output_path)})


def run_normalize_data() -> Dict[str, object]:
    """
    Exécute le traitement ML complet (chargement, fit/transform, sauvegarde
    CSV + préprocesseur) et retourne tout ce qui pourrait être nécessaire à
    des étapes ultérieures (ex : stockage Postgres), sans que cette fonction
    ait besoin de connaître Postgres.
    """
    logger.info({"event": "start_normalize_data", "seed": RANDOM_SEED})

    X_train, X_test = load_train_test()
    y_train, y_test = load_target()

    preprocessor, X_train_scaled, X_test_scaled, colonnes = fit_transform_preprocessor(
        X_train, X_test, y_train
    )

    save_scaled_datasets(X_train_scaled, X_test_scaled, colonnes)
    save_preprocessor(preprocessor)

    logger.info({
        "event": "end_normalize_data",
        "X_train_scaled_shape": list(X_train_scaled.shape),
        "X_test_scaled_shape": list(X_test_scaled.shape),
    })

    return {
        "preprocessor": preprocessor,
        "columns": list(colonnes),
        "X_train_scaled": pd.DataFrame(X_train_scaled, columns=colonnes),
        "X_test_scaled": pd.DataFrame(X_test_scaled, columns=colonnes),
        "y_train": y_train,
        "y_test": y_test,
    }


# ============================================================
# Stockage Postgres — fonction appelée APRÈS run_normalize_data()
# ============================================================

def _build_features_records(
    X_scaled: pd.DataFrame,
    y: Optional[pd.Series],
    dataset_split: str,
    run_id: str,
) -> pd.DataFrame:
    """Transforme un DataFrame de features scalées en lignes (run_id, split, row_index, features JSON, target)."""
    records = pd.DataFrame({
        "run_id": run_id,
        "dataset_split": dataset_split,
        "row_index": range(len(X_scaled)),
        "features": [json.dumps(row) for row in X_scaled.to_dict(orient="records")],
        "target": y.reset_index(drop=True) if y is not None else None,
    })

    return records


def save_ml_data_to_postgres(
    data: Dict[str, object],
    run_id: Optional[str] = None,
    connection_uri: Optional[str] = None,
    table_name: str = TABLE_FEATURES_ML,
):
    """
    À appeler APRÈS run_normalize_data(), sur le dict qu'elle retourne — ne
    modifie donc rien au comportement existant du traitement ML.

    Stocke data["X_train_scaled"] / data["X_test_scaled"] (features après
    imputation + scaling + encodage) dans la table Postgres weather_features_ml,
    en JSONB (1 ligne = 1 observation), avec la cible associée si disponible.

    """
    from sqlalchemy import create_engine
    from sqlalchemy.dialects.postgresql import JSONB

    required_keys = ("X_train_scaled", "X_test_scaled")
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(
            f"Clés manquantes {missing} : ce dict doit provenir de run_normalize_data()."
        )

    run_id = run_id or str(uuid.uuid4())

    train_records = _build_features_records(
        data["X_train_scaled"], data.get("y_train"), dataset_split="train", run_id=run_id
    )
    test_records = _build_features_records(
        data["X_test_scaled"], data.get("y_test"), dataset_split="test", run_id=run_id
    )
    all_records = pd.concat([train_records, test_records], ignore_index=True)

    if connection_uri is None:
        connection_uri = load_postgres_config().sqlalchemy_uri
    engine = create_engine(connection_uri)

    logger.info({
        "event": "inserting_ml_features_postgres",
        "table": table_name,
        "rows": len(all_records),
        "run_id": run_id,
    })

    all_records.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
        dtype={"features": JSONB},
    )

    logger.info({"event": "insert_completed", "table": table_name, "rows": len(all_records)})


def main() -> None:

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
    
    logger.info({"event": "normalize_data_start", "host": cfg.host, "db": cfg.db})
        
    result = run_normalize_data()

    try:
        logger.info({"event": "normalize_data_saving", "connection_uri": connstring, "table_name": SETTINGS["postgres"]["table_features"]})
     
        save_ml_data_to_postgres(
            result,
            run_id=str(uuid.uuid4()),
            connection_uri=connstring,
            table_name=SETTINGS["postgres"]["table_features"]
        )
    except Exception as e:
        logger.error({"event": "backup_ml_features_failed", "error": str(e)}, exc_info=True)


if __name__ == "__main__":
    main()
