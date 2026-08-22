#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Export du dataset WeatherAUS vers Parquet.
    
    Description :
        Le script extrait les données depuis :
            - CSV local
            - PostgreSQL
            - API REST
        Puis sauvegarde un snapshot Parquet utilisé par DVC = artefact de reproductibilité.

    Version :
        1.0.0

    Historique :
        2026-08-05  -  Création du module
===============================================================================
"""

import os
import sys

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np                  

from core.logger import get_logger
from core.settings import SETTINGS
from core.params import load_params
from core.metadata import NON_TECHNICAL_COLUMNS                                          


logger = get_logger("export_dataset")

OUTPUT_PATH = (Path(SETTINGS["paths"]["data"])/ SETTINGS["models"]["raw_dataset"])

# Code de sortie signalant à Airflow (DockerOperator.skip_on_exit_code) qu'il
# n'y a aucune donnée à traiter : la tâche et les tâches en aval doivent être
# marquées "skipped", pas "failed".
SKIP_EXIT_CODE = 99

# ============================================================
# Constantes
# ------------------------------------------------------------
# IMPORTANT : toutes les constantes métier (colonnes, contraintes,
# seuils, valeurs NA...) proviennent de settings.SETTINGS, qui est
# la source de vérité unique du projet (voir settings.py). Elles ne
# doivent JAMAIS être redéfinies ici.
# ============================================================

TARGET = SETTINGS["target"]["column_norm"]
LOCATION = SETTINGS["location"]["column_norm"]

 
# Valeurs considérées comme NA (CSV + Postgres)
POSTGRES_NA_VALUES = SETTINGS["na_values"]

PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
PREPROCESSED_DATA_PATH = PROCESSED_DIR / SETTINGS["models"]["preprocessed_data"]

IMPORT_DATE_COL = SETTINGS["postgres"]["importdate_column_norm"]  # "date_import"

TABLE_CLEAN = SETTINGS["postgres"]["table_clean"]
CLEAN_DATE_COL = SETTINGS["postgres"]["cleandate_column_norm"] 
RUNID_COL = SETTINGS["postgres"]["cleanrunid_column_norm"]
SOURCE_COL = SETTINGS["postgres"]["cleansource_column_norm"]



# ============================================================
# Chargement des données
# ============================================================

def load_data_from_csv(data_path: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un fichier CSV.

    Parameters
    ----------
    data_path : str
        Chemin vers le fichier CSV.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")

    logger.info("Chargement CSV depuis %s", data_path)
    
    df = pd.read_csv(path)

    return df


def load_data_from_parquet(data_path: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un fichier Parquet.

    Parameters
    ----------
    data_path : str
        Chemin vers le fichier Parquet.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")

    logger.info("Chargement Parquet depuis %s", data_path)

    return pd.read_parquet(path)


def load_data_from_postgres(connection_uri: str, table_name: str) -> pd.DataFrame:
    """
    Charge le dataset depuis une table PostgreSQL via SQLAlchemy.
    """

    try:
        from sqlalchemy import create_engine, inspect, text
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy est requis pour charger depuis PostgreSQL. "
            "Installez-le avec : pip install sqlalchemy psycopg2-binary"
        ) from exc

    logger.info(
        "Connexion PostgreSQL, lecture de la table '%s'",
        table_name
    )

    engine = create_engine(connection_uri)

    inspector = inspect(engine)

    if "." in table_name:
        schema, table = table_name.split(".")
    else:
        schema = None
        table = table_name

    if not inspector.has_table(table, schema=schema):
        raise ValueError(
            f"Table inexistante : {table_name}"
        )

    columns = ", ".join(NON_TECHNICAL_COLUMNS)

    query = text(
        f"""
        SELECT {columns}
        FROM {table_name}
        WHERE {IMPORT_DATE_COL} = (
            SELECT MAX({IMPORT_DATE_COL})
            FROM {table_name}
        )
        """
    )

    logger.info("Requête PostgreSQL : '%s'", query)

    df = pd.read_sql(query, engine)


    # ==========================================================
    # Normalisation des valeurs manquantes
    # ==========================================================

    logger.info({
        "event": "postgres_before_na_cleaning",
        "rain_tomorrow_values": (
            df["rain_tomorrow"]
            .value_counts(dropna=False)
            .to_dict()
            if "rain_tomorrow" in df.columns
            else None
        )
    })


    # Nettoyage uniquement des colonnes texte
    object_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in object_columns:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace(POSTGRES_NA_VALUES, pd.NA)
        )

        # Conversion pandas NA -> numpy NaN.
        # NB : `df.replace({pd.NA: np.nan})` ne fonctionne PAS de façon
        # fiable (pd.NA a une sémantique d'égalité spéciale : pd.NA == pd.NA
        # ne vaut pas True, donc .replace() ne le reconnaît pas comme une
        # valeur à remplacer). On repasse la colonne en dtype "object" et on
        # utilise .where()/.notna(), qui gèrent pd.NA correctement.
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)                                                                      


    logger.info({
        "event": "postgres_after_na_cleaning",
        "rain_tomorrow_values": (
            df["rain_tomorrow"]
            .value_counts(dropna=False)
            .to_dict()
            if "rain_tomorrow" in df.columns
            else None
        )
    })

    return df

def load_data_from_api(api_url: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un endpoint REST retournant du JSON.

    Deux formats de réponse supportés :
    - Liste JSON directe  : [{...}, {...}]
    - Objet avec clé data : {"data": [{...}, {...}], ...}

    Parameters
    ----------
    api_url : str
        URL complète de l'endpoint (ex. http://localhost:8057/meteo).

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    Dépendance requise : requests.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "requests est requis pour charger depuis l'API. "
            "Installez-le avec : pip install requests"
        ) from exc

    logger.info("Requête API : %s", api_url)
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    elif isinstance(payload, dict) and "data" in payload:
        df = pd.DataFrame(payload["data"])
    else:
        df = pd.DataFrame(payload)

    return df


def extract_dataset(
    source: str = "csv",
    data_path: Optional[str] = None,
    connection_uri: Optional[str] = None,
    table_name: Optional[str] = None,
    api_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Charge le dataset depuis la source spécifiée.

    Par défaut, utilise le CSV local pour préserver la compatibilité existante.

    Parameters
    ----------
    source : {"csv", "parquet", "postgres", "api"}
        Source de données. Par défaut : "csv".
    data_path : str, optional
        Chemin CSV ou Parquet. Requis si source="csv" ou source="parquet".
    connection_uri : str, optional
        URI SQLAlchemy. Requis si source="postgres".
    table_name : str, optional
        Nom de la table PostgreSQL. Requis si source="postgres".
    api_url : str, optional
        URL de l'endpoint REST. Requis si source="api".

    Returns
    -------
    pd.DataFrame
    """

    logger.info({
        "event": "loading data",
        "source": source
    })

    if source == "csv":
        if not data_path:
            raise ValueError("source='csv' requiert data_path.")
        return load_data_from_csv(data_path)

    if source == "parquet":
        if not data_path:
            raise ValueError("source='parquet' requiert data_path.")
        return load_data_from_parquet(data_path)

    if source == "postgres":
        if not connection_uri or not table_name:
            raise ValueError(
                "source='postgres' requiert connection_uri et table_name."
            )
        return load_data_from_postgres(connection_uri, table_name)

    if source == "api":
        if not api_url:
            raise ValueError("source='api' requiert api_url.")
        return load_data_from_api(api_url)

    raise ValueError(
        f"Source inconnue : '{source}'. Valeurs acceptées : csv, parquet, postgres, api."
    )


# ============================================================
# Export Parquet
# ============================================================

def save_parquet(
    df: pd.DataFrame,
    path: Path
):

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, engine="pyarrow", index=False)

    logger.info({
        "event": "parquet_saved",
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns)
    })


# ============================================================
# Main
# ============================================================

def main():
    
    logger.info(
        {
            "event": "export_dataset_start",
        }
    )
    PARAMS = load_params()
    SOURCE = PARAMS["dataset"]["source"]
    CSV_PATH = PARAMS["csv"]["path"]
    POSTGRES_TABLE = PARAMS["postgres"]["table"]
    API_URL = PARAMS["api"]["url"]

    connection_uri = os.environ.get("POSTGRES_URI")

    logger.info(
        {
            "event": "export_dataset_start",
            "source": SOURCE
        }
    )
    
    df = extract_dataset(
        source=SOURCE,
        data_path=CSV_PATH,
        connection_uri=connection_uri,
        table_name=POSTGRES_TABLE,
        api_url=API_URL,
    )

    if len(df) == 0:
        logger.warning(
            {
                "event": "export_dataset_skipped",
                "message": "Aucune ligne extraite, export Parquet annulé.",
                "source": SOURCE,
            }
        )
        print("⚠️  export_dataset SKIPPED (0 ligne, aucun fichier Parquet généré)")
        print(f"   Source  : {SOURCE}")
        sys.exit(SKIP_EXIT_CODE)

    save_parquet(df, OUTPUT_PATH)

    logger.info(
        {
            "event": "export_dataset_completed",
            "source": SOURCE,
            "rows": len(df),
            "columns": len(df.columns),
            "output": str(OUTPUT_PATH),
        }
    )

    print("✅ export_dataset OK")
    print(f"   Source  : {SOURCE}")
    print(f"   Output  : {OUTPUT_PATH}")
    print(f"   Shape   : {df.shape}")


if __name__ == "__main__":
    main()