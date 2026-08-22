"""
Module de prétraitement des données WeatherAUS.

Objectif :
Préparer les données pour prédire la variable cible RainTomorrow.

Ce module couvre :
1.  Chargement des données (CSV, PostgreSQL, API REST)
2.  Validation du schéma
3.  Rapport des valeurs manquantes
4.  Identification des colonnes avec plus de 30 % de valeurs manquantes
5.  Nettoyage et encodage de la cible
6.  Encodage de RainToday en binaire (0/1)
7.  Parsing de la date
8.  Feature engineering temporel
9.  Encodage cyclique (mois, jour de l'année, directions de vent)
10. Feature engineering météo
11. Séparation X / y (colonnes techniques exclues des features)
12. Split train/test
13. Construction du préprocesseur scikit-learn
14. Prévention de la fuite de données

Sources de données supportées :
- CSV local (défaut, rétrocompatible)
- PostgreSQL via SQLAlchemy (nécessite sqlalchemy, psycopg2-binary)
- API REST JSON (nécessite requests)

Stratégie de traitement des valeurs manquantes :
- Numériques  : IterativeImputer (modélise chaque variable à partir des autres)
                + indicateur de missingness + RobustScaler (robuste aux outliers)
- Location    : TargetEncoder (probabilité de pluie par ville, 1 colonne vs 49)
                Fallback vers OneHotEncoder si TargetEncoder indisponible (sklearn < 1.3).
- Catégorielles restantes : imputation mode + OneHotEncoder

Le préprocesseur est retourné non entraîné.
Il doit être intégré dans un Pipeline scikit-learn avec le modèle final.

Colonnes techniques (date_import, data_source, run_id) :
- Conservées dans les données brutes pour le monitoring Airflow.
- Exclues de X (features du modèle) par drop_unused_columns et split_features_target.
"""

import argparse
import logging
import os
import uuid

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.config import load_postgres_config
from core.logger import get_logger
from core.settings import SETTINGS
from core.metadata import COLUMN_CONSTRAINTS, REQUIRED_COLUMNS, TECHNICAL_COLUMNS, NON_TECHNICAL_COLUMNS, FEATURE_COLUMNS, WIND_DIRECTION_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, get_all_columns, normalize_column_name, normalize_data


# Les variables de threads doivent être posées AVANT l'import de numpy /
# pandas / sklearn pour avoir un effet (BLAS/OMP les lit à l'import).
for _env_key, _env_val in SETTINGS["threads"].items():
    os.environ.setdefault(_env_key, str(_env_val))

import numpy as np
import pandas as pd

logger = get_logger("build_features")


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
 
HIGH_MISSING_THRESHOLD = SETTINGS["missing_threshold"]
SPLIT_STRATEGY = SETTINGS["split_strategy"]

# Degrés pour chaque direction cardinale, N = 0°, sens horaire.
COMPASS_DEGREES: Dict[str, float] = SETTINGS["compass_degrees"]
 
# Valeurs considérées comme NA (CSV + Postgres)
POSTGRES_NA_VALUES = SETTINGS["na_values"]

REPORTS_DIR = Path(SETTINGS["paths"]["reports"])
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

    # Normalisation des noms de colonne
    return normalize_data(df)


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


    # Normalisation des noms de colonne
    df = normalize_data(df)


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

    # Normalisation des noms de colonne
    return normalize_data(df)


def load_dataset(
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
    source : {"csv", "postgres", "api"}
        Source de données. Par défaut : "csv".
    data_path : str, optional
        Chemin CSV. Requis si source="csv".
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
        f"Source inconnue : '{source}'. Valeurs acceptées : csv, postgres, api."
    )

    
# ============================================================
# Validation du schéma
# ============================================================

def validate_schema(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> None:
    """
    Vérifie que le dataset contient toutes les colonnes attendues.

    Raises
    ------
    ValueError
        Si une ou plusieurs colonnes obligatoires sont absentes.
    """
    expected_columns = columns or REQUIRED_COLUMNS
    
    # Détection des colonnes manquantes
    missing_columns = sorted(set(expected_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(
            "Le dataset ne respecte pas le schéma attendu. "
            f"Colonnes manquantes : {missing_columns}"
            f"Colonnes reçues : {list(df.columns)}"
        )


# ============================================================
# Convertion du type des colonnes
# ============================================================

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit automatiquement les colonnes selon les contraintes définies
    dans COLUMN_CONSTRAINTS :
    - type : float, datetime, string
    - nullable : True/False
    - allowed_values : liste de valeurs autorisées
    - range : (min, max)
    """

    df = df.copy()

    for col, meta in COLUMN_CONSTRAINTS.items():
     
        # Si la colonne n'existe pas, on logge et on continue
        if meta["norm_name"] not in df.columns:
            logger.warning(f"Colonne absente dans le DataFrame : {col}/{meta['norm_name']}")
            continue

        col_type = meta.get("type")
        col_norm = meta["norm_name"]

        # --- Conversion float ---
        if col_type == "float":
            df[col_norm] = pd.to_numeric(df[col_norm], errors="coerce")

            # Vérification des valeurs hors plage
            if "range" in meta:
                min_val, max_val = meta["range"]
                out_of_range = df[(df[col_norm] < min_val) | (df[col_norm] > max_val)]
                if not out_of_range.empty:
                    logger.warning(
                        f"Valeurs hors plage détectées dans {col} "
                        f"({min_val} → {max_val}). "
                        f"Lignes concernées : {len(out_of_range)}"
                    )

        # --- Conversion datetime ---
        elif col_type == "datetime":
            df[col_norm] = pd.to_datetime(df[col_norm], errors="coerce")

        # --- Conversion string ---
        elif col_type == "string":
            df[col_norm] = df[col_norm].astype(object)

            # Vérification des valeurs autorisées
            allowed = meta.get("allowed_values")
            if allowed is not None:
                invalid = df[df[col_norm].notna() & ~df[col_norm].isin(allowed)]
                if not invalid.empty:
                    logger.warning(
                        f"Valeurs invalides détectées dans {col}. "
                        f"Modalités autorisées : {allowed}. "
                        f"Lignes concernées : {len(invalid)}"
                    )

        # --- Vérification nullable ---
        if meta.get("nullable") is False:
            if df[col_norm].isna().any():
                missing = df[df[col_norm].isna()]
                logger.error(
                    f"Colonne {col} ne doit pas contenir de valeurs manquantes. "
                    f"Lignes concernées : {len(missing)}"
                )
                raise ValueError(f"Valeurs manquantes détectées dans {col}, qui est non-nullable.")

    return df


# ============================================================
# Sauvegarde des données nettoyées vers PostgreSQL (table clean)
# ============================================================

def build_postgres_engine(connection_uri: Optional[str] = None):
    """
    Construit un engine SQLAlchemy.

    Si connection_uri n'est pas fourni, utilise la configuration Postgres
    du projet (variables d'environnement, voir config.load_postgres_config).
    """
    from sqlalchemy import create_engine

    if connection_uri is None:
        connection_uri = load_postgres_config().sqlalchemy_uri

    return create_engine(connection_uri)


def add_technical_columns(
    df: pd.DataFrame,
    data_source: str,
    run_id: str,
) -> pd.DataFrame:
    """
    Ajoute les colonnes techniques d'audit (date_import, data_source, run_id)
    définies dans metadata.TECHNICAL_COLUMNS, utilisées pour le monitoring
    Airflow et pour ne lire que le dernier import (voir load_data_from_postgres).
    """
    df = df.copy()

    df[CLEAN_DATE_COL] = pd.Timestamp.now(tz="UTC")
    df[SOURCE_COL] = data_source
    df[RUNID_COL] = run_id

    return df


def save_clean_data_to_csv(
    df: pd.DataFrame
) -> Path:
    """
    Sauvegarde les données nettoyées dans un fichier CSV.

    Si le fichier existe déjà, les nouvelles lignes sont ajoutées à la suite
    (mode append, cohérent avec if_exists="append" côté Postgres) ; sinon le
    fichier est créé avec l'en-tête.

    Parameters
    ----------
    df : pd.DataFrame
        Données déjà nettoyées et enrichies des colonnes techniques
        (date_import, data_source, run_id).
    
    Returns
    -------
    Path
        Chemin du fichier écrit.
    """
    PREPROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_exists = PREPROCESSED_DATA_PATH.exists()
    df.to_csv(PREPROCESSED_DATA_PATH, mode="a", header=not file_exists, index=False)

    logger.info({
        "event": "clean_data_saved_csv",
        "path": str(PREPROCESSED_DATA_PATH),
        "rows": len(df),
        "mode": "append" if file_exists else "create",
    })

    return PREPROCESSED_DATA_PATH

def save_prepared_data(
    data: pd.DataFrame,
    source: str,
    run_id: Optional[str] = None,
    connection_uri: Optional[str] = None,
    table_name: Optional[str] = None
):
    """
    Insère data["df_clean"] (données nettoyées / typées, post convert_types,
    AVANT tout feature engineering ML) dans la table Postgres "clean"
    (weather_data_clean par défaut). Ne fait aucun encodage / scaling.

   """
    run_id = run_id or str(uuid.uuid4())

    ordered_columns = list(FEATURE_COLUMNS)

    missing = [c for c in ordered_columns if c not in data.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes avant sauvegarde : {missing}")

    df_to_insert = add_technical_columns(data, data_source=source, run_id=run_id)

    logger.info({
        "event": "saving_clean_data",
        "destination": source,
        "rows": len(df_to_insert),
        "run_id": run_id
    })               
    if source == "csv":
        save_clean_data_to_csv(df_to_insert)

    elif source == "postgres":
        logger.info({
            "event": "saving_clean_data_postgres",
            "table_name": table_name,
            "connection": connection_uri
        })   
        
        required_params = {
            "table_name": table_name,
            "connection_uri": connection_uri,
        }
        missing = [
            name for name, value in required_params.items()
            if not value
        ]
        if missing:
            raise ValueError(
                f"Paramètre(s) obligatoire(s) non fourni(s) : {', '.join(missing)}"
            )
        
        engine = None
        try:
            engine = build_postgres_engine(connection_uri)

            with engine.begin() as connection:
                df_to_insert.to_sql(
                    table_name,
                    connection,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=1000
                )

        except Exception:
            logger.error("Erreur lors de l'insertion PostgreSQL")
            raise
        finally:
            if engine:
                engine.dispose()
                
    elif source == "api":
        ### meme stockage que csv
        save_clean_data_to_csv(df_to_insert)
    else:
        raise ValueError(
            f"Source inconnue : '{source}'. Valeurs acceptées : csv, postgres, api."
        )  
    
    logger.info({"event": "save_completed", "destination": source, "rows": len(df_to_insert), "run_id": run_id})
     

# ============================================================
# Rapport des valeurs manquantes
# ============================================================

def build_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport détaillé sur les valeurs manquantes.

    Returns
    -------
    pd.DataFrame
        Rapport contenant le taux, le nombre de valeurs manquantes,
        le type de variable et le nombre de valeurs distinctes.
    """
    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
            "missing_rate_pct": (df.isna().mean().values * 100).round(2),
            "unique_values": df.nunique(dropna=True).values,
        }
    )

    report = report.sort_values("missing_rate", ascending=False)
    report = report.reset_index(drop=True)

    return report


def save_missing_values_report(
    report: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> None:
    
    """Sauvegarde le rapport des valeurs manquantes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / SETTINGS["models"]["reports"]

    report.to_csv(output_file, index=False)

    logger.info({"event": "missing_values_report_saved", "path": str(output_file)})

def identify_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD,
    exclude_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Identifie les colonnes dont le taux de valeurs manquantes dépasse le seuil.

    Par défaut, le seuil est 30 %.

    Important :
    Ces colonnes ne sont pas supprimées automatiquement.
    La stratégie principale consiste à les conserver avec imputation et
    indicateur de valeur manquante.
    """
    if exclude_columns is None:
        exclude_columns = []

    missing_rate = df.isna().mean()

    high_missing_columns = (
        missing_rate[missing_rate > threshold]
        .index
        .difference(exclude_columns)
        .tolist()
    )

    return high_missing_columns


# ============================================================
# Nettoyage de la cible
# ============================================================

def clean_target(
    df: pd.DataFrame,
    target: Optional[str] = TARGET
) -> pd.DataFrame:
    """
    Nettoie et encode la variable cible RainTomorrow.

    Traitements :
    - suppression des lignes où RainTomorrow est manquante ;
    - encodage binaire : No = 0, Yes = 1.
    """
    
    logger.info({
            "event": "starting clean-target",
            "target": target
    })

    target = normalize_column_name(target)

    logger.info({
            "event": "normalize target",
            "target": target
    })
    
    df = df.copy() 

    logger.info(
        "POSTGRES_NA_VALUES=%s",
        POSTGRES_NA_VALUES
    )
    logger.info(
        "Modalités RainTomorrow avant drop : %s",
        df[target].value_counts(dropna=False).to_dict()
    )
    df = df.dropna(subset=[target])
    
    logger.info(
        "Modalités RainTomorrow après drop mapping : %s",
        df[target].value_counts(dropna=False).to_dict()
    )
    
    target_mapping = {"No": 0, "Yes": 1}
    df[target] = df[target].map(target_mapping)
    
    logger.info(
        "Modalités RainTomorrow après mapping : %s",
        df[target].value_counts(dropna=False).to_dict()
    )

    if df[target].isna().any():
        
        missing_idx = df[df[target].isna()].index.tolist()

        # Aperçu des lignes fautives (limité à 5 pour éviter les logs énormes)
        sample_rows = df.loc[missing_idx].head(5)

        logger.error(
            "La variable cible contient des valeurs manquantes. "
            f"Nombre de lignes concernées : {len(missing_idx)}. "
            f"Index des premières lignes : {missing_idx[:10]}"
        )

        logger.error("Aperçu des lignes concernées :\n%s", sample_rows)
    
        raise ValueError(
            "La variable cible contient des modalités inattendues. "
            "Modalités attendues : 'Yes' et 'No'."
        )

    df[target] = df[target].astype(int)

    return df


# ============================================================
# Encodage des features
# ============================================================    
    
def encode_rain_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode RainToday en binaire (0/1) pour cohérence avec RainTomorrow.

    Les valeurs manquantes restent NaN et seront traitées par l'imputer.
    Evite les 2 colonnes OHE et traite RainToday comme une variable numérique.
    """
    df = df.copy()
    
    raintoday = normalize_column_name("RainToday")
    df[raintoday] = df[raintoday].map({"No": 0, "Yes": 1})
    return df


# ============================================================
# Traitement de la date
# ============================================================

import re
import pandas as pd


def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit la colonne Date en datetime.

    Le format attendu est YYYY-MM-DD.
    Les dates invalides sont transformées en NaT puis supprimées.
    """
    df = df.copy()

    date_norm = normalize_column_name("Date")

    # Vérifie le format avant conversion
    valid_format = df[date_norm].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")

    if (~valid_format).any():
        invalid_rows = df.loc[~valid_format, [date_norm]]

        logger.warning(
            "Format de date invalide détecté (%s lignes). Exemples : %s",
            len(invalid_rows),
            invalid_rows.head(10).to_dict("records"),
        )

    # Conversion en datetime
    df[date_norm] = pd.to_datetime(
        df[date_norm],
        format="%Y-%m-%d",
        errors="coerce",
    )

    # Suppression des dates invalides
    before = len(df)
    df = df.dropna(subset=[date_norm])

    logger.info(
        "Suppression de %s lignes contenant une date invalide.",
        before - len(df),
    )

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables temporelles à partir de Date.

    La colonne Date doit être convertible en datetime.
    Les dates invalides sont supprimées.
    """
    df = df.copy()

    date_norm = normalize_column_name("Date")

    # Vérification présence colonne
    if date_norm not in df.columns:
        raise ValueError(
            f"Impossible de créer les variables temporelles : "
            f"colonne '{date_norm}' absente."
        )

    # Vérification / conversion du type
    if not pd.api.types.is_datetime64_any_dtype(df[date_norm]):
        logger.warning(
            "La colonne %s n'est pas au format datetime. Conversion forcée.",
            date_norm
        )

        df[date_norm] = pd.to_datetime(
            df[date_norm],
            format="%Y-%m-%d",
            errors="coerce"
        )

    # Vérification des dates invalides
    invalid_dates = df[date_norm].isna()

    if invalid_dates.any():
        count = invalid_dates.sum()
        logger.error(
            "Dates invalides détectées dans %s : %s lignes supprimées",
            date_norm,
            count
        )

        df = df.dropna(subset=[date_norm])


    # Contrôle avant création des features temporelles
    if df.empty:
        raise ValueError(
            "Impossible de créer les features temporelles : "
            "le DataFrame ne contient aucune ligne."
        )

    if date_norm not in df.columns:
        raise ValueError(
            f"Impossible de créer les features temporelles : "
            f"colonne '{date_norm}' absente."
        )

    if not pd.api.types.is_datetime64_any_dtype(df[date_norm]):
        raise TypeError(
            f"Impossible de créer les features temporelles : "
            f"la colonne '{date_norm}' n'est pas au format datetime "
            f"(type détecté : {df[date_norm].dtype})."
        )

    if df[date_norm].isna().any():
        nb_dates_invalides = df[date_norm].isna().sum()
        raise ValueError(
            f"Impossible de créer les features temporelles : "
            f"{nb_dates_invalides} date(s) invalide(s) détectée(s)."
        )

    # Création des features temporelles
    df[normalize_column_name("Year")] = df[date_norm].dt.year
    df[normalize_column_name("Month")] = df[date_norm].dt.month
    df[normalize_column_name("Day")] = df[date_norm].dt.day
    df[normalize_column_name("DayOfYear")] = df[date_norm].dt.dayofyear

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute un encodage cyclique pour Month et DayOfYear.

    Cet encodage permet de mieux représenter la saisonnalité.
    Par exemple, décembre et janvier sont proches dans le temps.
    """
    df = df.copy()

    month_norm = normalize_column_name("Month")
    doy_norm = normalize_column_name("DayOfYear")
    df[normalize_column_name("Month_sin")] = np.sin(2 * np.pi * df[month_norm] / 12)
    df[normalize_column_name("Month_cos")] = np.cos(2 * np.pi * df[month_norm] / 12)

    df[normalize_column_name("DayOfYear_sin")] = np.sin(2 * np.pi * df[doy_norm] / 365)
    df[normalize_column_name("DayOfYear_cos")] = np.cos(2 * np.pi * df[doy_norm] / 365)

    return df


def encode_wind_directions(
    df: pd.DataFrame,
    wind_direction_columns: Optional[List[str]] = WIND_DIRECTION_COLUMNS,
) -> pd.DataFrame:
    """
    Convertit les colonnes de direction de vent en encodage cyclique sin/cos.

    Les 16 directions cardinales (N, NNE, …, NNW) sont mappées en degrés
    puis projetées sur sin/cos pour préserver la continuité circulaire :
    N (0°) et NNW (337.5°) sont voisins, ce qu'OHE ne peut pas capturer.
    Les colonnes string originales sont supprimées après transformation.
    """
    wind_direction_columns = [normalize_column_name(c) for c in wind_direction_columns]

    df = df.copy()

    for col in wind_direction_columns:
        if col not in df.columns:
            continue

        degrees = df[col].map(COMPASS_DEGREES)
        df[f"{col}_sin"] = np.sin(np.radians(degrees))
        df[f"{col}_cos"] = np.cos(np.radians(degrees))
        df = df.drop(columns=[col])

    return df


# ============================================================
# Feature engineering météo
# ============================================================

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables météo supplémentaires.

    Variables créées :
    - TempRange
    - TempChange
    - HumidityDrop
    - PressureDrop
    - WindSpeedChange
    - RainfallLog1p
    - HasRainfall
    - HighHumidity3pm
    - StrongWindGust
    """
    df = df.copy()

    df[normalize_column_name("TempRange")] = df[normalize_column_name("MaxTemp")] - df[normalize_column_name("MinTemp")]
    df[normalize_column_name("TempChange")] = df[normalize_column_name("Temp3pm")] - df[normalize_column_name("Temp9am")]

    df[normalize_column_name("HumidityDrop")] = df[normalize_column_name("Humidity9am")] - df[normalize_column_name("Humidity3pm")]
    df[normalize_column_name("PressureDrop")] = df[normalize_column_name("Pressure9am")] - df[normalize_column_name("Pressure3pm")]
    df[normalize_column_name("WindSpeedChange")] = df[normalize_column_name("WindSpeed3pm")] - df[normalize_column_name("WindSpeed9am")]

    df[normalize_column_name("RainfallLog1p")] = np.log1p(df[normalize_column_name("Rainfall")].clip(lower=0))

    hrf_norm = normalize_column_name("HasRainfall")
    rainfall_norm = normalize_column_name("Rainfall")
    df[hrf_norm] = np.where(
        df[rainfall_norm].isna(),
        np.nan,
        (df[rainfall_norm] > 0).astype(int),
    )

    hh3pm_norm = normalize_column_name("HighHumidity3pm")
    hum3pm_norm = normalize_column_name("Humidity3pm")
    df[hh3pm_norm] = np.where(
        df[hum3pm_norm].isna(),
        np.nan,
        (df[hum3pm_norm] >= 70).astype(int),
    )

    swg_norm = normalize_column_name("StrongWindGust")
    wgs_norm = normalize_column_name("WindGustSpeed")
    df[swg_norm] = np.where(
        df[wgs_norm].isna(),
        np.nan,
        (df[wgs_norm] >= 50).astype(int),
    )

    return df


# ============================================================
# Gestion des colonnes fortement manquantes
# ============================================================

def drop_high_missing_columns(
    df: pd.DataFrame,
    high_missing_columns: List[str],
    protected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Supprime les colonnes fortement incomplètes si le scénario est activé.

    Par défaut, on conserve ces colonnes.
    Cette fonction sert uniquement à tester un scénario alternatif.
    """
    df = df.copy()

    if protected_columns is None:
        protected_columns = [normalize_column_name(TARGET), normalize_column_name("Date")]

    columns_to_drop = [
        col for col in high_missing_columns
        if col in df.columns and col not in protected_columns
    ]

    return df.drop(columns=columns_to_drop)



def drop_unused_columns(
    df: pd.DataFrame,
    technical_columns: Optional[List[str]] = TECHNICAL_COLUMNS,
) -> pd.DataFrame:
    """
    Supprime les colonnes inutiles après feature engineering.

    Colonnes supprimées si présentes :
    - Date         : extraite en variables temporelles (Year, Month, DayOfYear, …)
    - date_import  : colonne technique Airflow, hors features modèle
    - data_source  : colonne technique Airflow, hors features modèle
    - run_id       : colonne technique Airflow, hors features modèle
    """
    technical_columns = [normalize_column_name(c) for c in technical_columns]

    df = df.copy()

    columns_to_drop = ["date"] + technical_columns
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    return df.drop(columns=existing_columns)


# ============================================================
# Séparation X / y
# ============================================================

def split_features_target(
    df: pd.DataFrame,
    target: Optional[str] = TARGET,
    technical_columns: Optional[List[str]] = TECHNICAL_COLUMNS
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les variables explicatives X et la cible y.

    Les colonnes techniques (TECHNICAL_COLUMNS) sont exclues de X
    si elles n'ont pas déjà été supprimées par drop_unused_columns.
    """
    
    target = normalize_column_name(target)
    technical_columns = [normalize_column_name(c) for c in technical_columns]
            
    cols_to_exclude = [target] + [
        c for c in technical_columns if c in df.columns
    ]
    X = df.drop(columns=cols_to_exclude)
    y = df[target]

    return X, y


# ============================================================
# Identification des types de variables
# ============================================================

def identify_feature_types(
    X: pd.DataFrame
) -> Tuple[List[str], List[str]]:
    """Identifie les variables numériques et catégorielles à partir du contrat de données du DF."""
    
    numeric_features = NUMERIC_COLUMNS

    categorical_features = CATEGORICAL_COLUMNS

    return numeric_features, categorical_features


# ============================================================
# Préprocesseur scikit-learn
# ============================================================

def make_one_hot_encoder():
    """
    Crée un OneHotEncoder compatible avec plusieurs versions de scikit-learn.

    L'import est effectué localement pour permettre de tester les étapes
    pandas même si scikit-learn / SciPy pose problème dans l'environnement.
    """
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
):
    """
    Construit le préprocesseur complet (non entraîné).

    Tous les imports sklearn sont effectués localement pour permettre de
    tester les étapes pandas même si scikit-learn / SciPy pose problème.

    Variables numériques :
    - IterativeImputer : modélise chaque variable manquante à partir de toutes
      les autres via régression bayésienne, préservant les corrélations naturelles
      (ex : Temp9am / MaxTemp) — évite le biais d'une valeur unique ;
    - indicateurs de valeurs manquantes (add_indicator=True) ;
    - RobustScaler : utilise médiane + IQR, insensible aux outliers de Rainfall
      et WindGustSpeed contrairement à StandardScaler.

    Variable Location :
    - imputation par la modalité la plus fréquente ;
    - TargetEncoder si disponible (scikit-learn >= 1.3) : encode chaque ville
      par sa probabilité de pluie historique, réduisant 49 colonnes OHE à 1.
      Fallback vers OneHotEncoder sinon.

    Autres variables catégorielles :
    - imputation par la modalité la plus fréquente ;
    - encodage One-Hot.

    Le préprocesseur doit être intégré dans un Pipeline avec le modèle
    et fitté uniquement sur X_train.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    add_indicator=True,
                ),
            ),
            ("scaler", RobustScaler()),
        ]
    )

    # TargetEncoder disponible depuis scikit-learn 1.3
    try:
        from sklearn.preprocessing import TargetEncoder
        location_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", TargetEncoder(target_type="binary")),
            ]
        )
        logger.info("Encodage Location : TargetEncoder")
    except ImportError:
        location_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_one_hot_encoder()),
            ]
        )
        logger.warning(
            "TargetEncoder indisponible (scikit-learn < 1.3). "
            "Fallback vers OneHotEncoder pour Location."
        )

    other_cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    loc_norm = normalize_column_name("Location")
    location_features = [col for col in categorical_features if col == loc_norm]
    other_cat_features = [col for col in categorical_features if col != loc_norm]

    transformers: List = [("numeric", numeric_pipeline, numeric_features)]

    if location_features:
        transformers.append(("location", location_pipeline, location_features))

    if other_cat_features:
        transformers.append(("categorical", other_cat_pipeline, other_cat_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ============================================================
# Préparation déterministe
# ============================================================

def prepare_dataframe(
    df: pd.DataFrame,
    target: str,
    technical_columns: List[str],
    high_missing_columns: Optional[List[str]] = None,
    drop_high_missing: bool = False
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Applique les transformations déterministes.

    Cette fonction ne fait pas :
    - d'imputation statistique ;
    - de standardisation ;
    - d'encodage One-Hot ou target encoding.

    Ces opérations restent dans le ColumnTransformer pour éviter la fuite de données.

    Prévention de la fuite de données (data leakage)
    -------------------------------------------------
    Seules les transformations « sans mémoire » sont appliquées ici :
    encodage binaire de la cible, parsing de date, feature engineering
    arithmétique. Toute transformation qui apprend une statistique sur
    les données (moyenne, médiane, probabilités de pluie par ville…)
    est déléguée au ColumnTransformer, qui ne sera fitté que sur X_train
    à l'intérieur d'un Pipeline sklearn. Ainsi le test set reste un
    témoin strictement indépendant.
    """
    df = clean_target(df, target=target)

    df = encode_rain_today(df)
    df = parse_date_column(df)

    df_with_date = df.copy()

    df = add_temporal_features(df)
    df = add_cyclical_features(df)
    df = encode_wind_directions(df)
    df = add_weather_features(df)

    if drop_high_missing and high_missing_columns is not None:
        df = drop_high_missing_columns(
            df=df,
            high_missing_columns=high_missing_columns,
            protected_columns=[target, normalize_column_name("date")],
        )

    df = drop_unused_columns(df, technical_columns=technical_columns)

    X, y = split_features_target(df, target=target, technical_columns=technical_columns)

    return X, y, df_with_date


# ============================================================
# Split train/test
# ============================================================

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    df_for_sort: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = SPLIT_STRATEGY,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare les données en train/test.

    L'import sklearn est effectué localement pour permettre de tester les
    étapes pandas même si scikit-learn / SciPy pose problème.

    Stratégies disponibles :
    - random    : split aléatoire stratifié — acceptable pour un benchmark rapide.
    - temporal  : split chronologique (toutes les dates d'entraînement précèdent
                  celles du test) — obligatoire en production pour simuler la
                  prédiction sur des jours futurs et éviter le look-ahead leakage :
                  avec un split aléatoire, des observations d'un même lieu et
                  d'une même période peuvent se retrouver simultanément dans le
                  train ET le test, gonflant artificiellement les métriques.
    """
    if split_strategy not in ["random", "temporal"]:
        raise ValueError("split_strategy doit être 'random' ou 'temporal'.")

    if split_strategy == "random":
        from sklearn.model_selection import train_test_split
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    date_norm = normalize_column_name("Date")
    if df_for_sort is None or date_norm not in df_for_sort.columns:
        raise ValueError(
            "Pour split_strategy='temporal', df_for_sort doit contenir Date."
        )

    ordered_index = df_for_sort.sort_values(date_norm).index

    X_ordered = X.loc[ordered_index]
    y_ordered = y.loc[ordered_index]

    split_position = int(len(X_ordered) * (1 - test_size))

    X_train = X_ordered.iloc[:split_position]
    X_test = X_ordered.iloc[split_position:]
    y_train = y_ordered.iloc[:split_position]
    y_test = y_ordered.iloc[split_position:]

    return X_train, X_test, y_train, y_test


# ============================================================
# Fonction principale
# ============================================================

def prepare_data(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = SPLIT_STRATEGY,
    missing_threshold: float = HIGH_MISSING_THRESHOLD,
    drop_high_missing: bool = False,
    save_report: bool = True,
    source: str = "csv",
    connection_uri: Optional[str] = None,
    table_name: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict[str, object]:
    """
    Prépare les données pour la modélisation.

    Supporte trois sources : CSV (défaut, rétrocompatible), PostgreSQL, API REST.

    Par défaut :
    - les colonnes avec plus de 30 % de valeurs manquantes sont conservées ;
    - elles sont traitées par IterativeImputer + indicateur de missingness ;
    - la suppression est disponible comme scénario alternatif (drop_high_missing=True).

    Les colonnes techniques (date_import, data_source, run_id) provenant de
    PostgreSQL sont exclues des features du modèle mais disponibles dans les
    données brutes pour le monitoring Airflow.

    Parameters
    ----------
    data_path : str, optional
        Chemin CSV. Utilisé si source="csv".
    source : {"csv", "postgres", "api"}
        Source de données. Par défaut : "csv".
    connection_uri : str, optional
        URI SQLAlchemy. Requis si source="postgres".
    table_name : str, optional
        Nom de la table. Requis si source="postgres".
    api_url : str, optional
        URL de l'endpoint. Requis si source="api".
    """
    
    run_id = str(uuid.uuid4())
    
    logger.info({
        "event": "init_data_preprocessing",
        "source": source,
        "data_path": data_path,
        "connection_uri": connection_uri,
        "table_name": table_name,
        "api_url": api_url
    })
    
    if split_strategy is None:
        split_strategy = SPLIT_STRATEGY
    
    if split_strategy not in ["random", "temporal"]:
        raise ValueError("split_strategy doit être 'random' ou 'temporal'.")
        
    raw_df = load_dataset(
        source=source,
        data_path=data_path,
        connection_uri=connection_uri,
        table_name=table_name,
        api_url=api_url,
    )

    logger.info({
        "event": "normalizing_columns"
    })
    
    df_norm = normalize_data(raw_df)

    # Version normalisée du target
    target_norm = normalize_column_name(TARGET)
    
    logger.info({
        "event": "schema_data_validating"
    })
    
    validate_schema(df_norm, REQUIRED_COLUMNS)

    logger.info({
        "event": "converting_types_data"
    })
    
    df_norm_conv = convert_types(df_norm)

    missing_report = build_missing_values_report(df_norm_conv)

    high_missing_columns = identify_high_missing_columns(
        df=df_norm_conv,
        threshold=missing_threshold,
        exclude_columns=[target_norm],
    )

    if save_report:
        save_missing_values_report(missing_report)

    X, y, df_with_date = prepare_dataframe(
        df=df_norm_conv,
        target=target_norm,
        technical_columns= TECHNICAL_COLUMNS,
        high_missing_columns=high_missing_columns,
        drop_high_missing=drop_high_missing
    )

    LOCATION = normalize_column_name("Location")

    known_locations = sorted(
        X[LOCATION]
        .dropna()
        .unique()
        .tolist()
    )

    logger.info({
        "event": "known_locations_detected",
        "count": len(known_locations),
        "locations": known_locations
    })

    numeric_features, categorical_features = identify_feature_types(X)

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X=X,
        y=y,
        df_for_sort=df_with_date,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )

    
    logger.info({
        "event": "end_data_preprocessing",
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "y_train_distribution": y_train.value_counts(normalize=True).to_dict(),
        "y_test_distribution": y_test.value_counts(normalize=True).to_dict(),
        "numeric_features_count": len(numeric_features),
        "categorical_features_count": len(categorical_features),
        "high_missing_columns": high_missing_columns
        })
    
    try:
        logger.info("Enregistrement des données nettoyées")
        save_prepared_data(
                data=df_norm_conv,
                source=source,
                run_id=run_id,
                connection_uri=connection_uri,
                table_name=TABLE_CLEAN
            )
        logger.info("Fin enregistrement des données nettoyées")
        
    except Exception as e:
        logger.error({
            "event": "erreur_",
            "error": str(e),
            "run_id": run_id
        })
        
        
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_report": missing_report,
        "high_missing_columns": high_missing_columns,
        "feature_schema": {
            "numeric": numeric_features,
            "categorical": categorical_features
        },
        "data_version": {
            "rows": len(raw_df),
            "columns": len(raw_df.columns)
        }
    }


# ============================================================
# Tests locaux
# ============================================================

def check_preprocessor_works(data: Dict[str, object]) -> None:
    """
    Vérifie que le préprocesseur fonctionne correctement.

    Anti-leakage : fit_transform sur X_train uniquement, puis transform sur X_test.
    IterativeImputer apprend les distributions sur le train ; TargetEncoder calcule
    les probabilités de pluie par ville sur le train ; RobustScaler calcule médiane
    et IQR sur le train. Aucune statistique du test set ne remonte vers le train.

    En production, sauvegarder le preprocessor fitté (joblib.dump) comme artefact
    versionné afin que l'inférence applique exactement les mêmes transformations
    sans re-fitter — évite la dérive de transformation entre entraînement et serving.
    """
    preprocessor = data["preprocessor"]

    X_train_transformed = preprocessor.fit_transform(
        data["X_train"], data["y_train"]
    )
    X_test_transformed = preprocessor.transform(data["X_test"])

    logger.info("Test du préprocesseur : OK")
    logger.info("X_train transformé : %s", X_train_transformed.shape)
    logger.info("X_test transformé  : %s", X_test_transformed.shape)


def display_preprocessing_summary(data: Dict[str, object]) -> None:
    """Affiche un résumé du prétraitement."""
    logger.info("Prétraitement terminé.")
    logger.info("-" * 70)

    logger.info("X_train : %s", data["X_train"].shape)
    logger.info("X_test  : %s", data["X_test"].shape)
    logger.info("y_train : %s", data["y_train"].shape)
    logger.info("y_test  : %s", data["y_test"].shape)

    logger.info(
        "Distribution de y_train :\n%s",
        data["y_train"].value_counts(normalize=True).round(4),
    )
    logger.info(
        "Distribution de y_test :\n%s",
        data["y_test"].value_counts(normalize=True).round(4),
    )

    logger.info(
        "Colonnes avec plus de 30 %% de valeurs manquantes : %s",
        data["high_missing_columns"],
    )

    logger.info("Nombre de variables numériques : %d", len(data["numeric_features"]))
    logger.info("%s", data["numeric_features"])

    logger.info(
        "Nombre de variables catégorielles : %d",
        len(data["categorical_features"]),
    )
    logger.info("%s", data["categorical_features"])

    logger.info(
        "Top 10 des variables avec valeurs manquantes :\n%s",
        data["missing_report"].head(10),
    )

    logger.info("Préprocesseur :\n%s", data["preprocessor"])


def parse_arguments() -> argparse.Namespace:
    """Gère les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Prétraitement du dataset WeatherAUS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="csv",
        choices=["csv", "postgres", "api"],
        help="Source de données : csv (défaut), postgres ou api.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Chemin vers weatherAUS.csv (source=csv). "
        ),
    )

    parser.add_argument(
        "--connection-uri",
        type=str,
        default=None,
        help=(
            "URI de connexion PostgreSQL (source=postgres). "
            "Exemple : postgresql://USER:PWD@localhost:5432/DB_NAME"
        ),
    )

    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="Nom de la table PostgreSQL (source=postgres).",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=(
            "URL de l'endpoint API (source=api). "
            "Exemple : http://localhost:8057/meteo"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de test.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Graine de reproductibilité.",
    )

    parser.add_argument(
        "--split-strategy",
        type=str,
        default="temporal",
        choices=["random", "temporal"],
        help="Stratégie de split : random ou temporal.",
    )

    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=HIGH_MISSING_THRESHOLD,
        help="Seuil de valeurs manquantes. Exemple : 0.30 pour 30 %%.",
    )

    parser.add_argument(
        "--drop-high-missing",
        action="store_true",
        help="Supprime les colonnes avec trop de valeurs manquantes.",
    )

    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Désactive la sauvegarde du rapport des valeurs manquantes.",
    )

    parser.add_argument(
        "--save-clean-to-postgres",
        action="store_true",
        help=(
            "Insère les données nettoyées / typées (avant feature engineering ML) "
            "dans la table Postgres clean, juste après convert_types."
        ),
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifiant de run stocké dans la colonne technique run_id (généré si absent).",
    )
    parser.add_argument(
        "--clean-connection-uri",
        type=str,
        default=None,
        help="URI SQLAlchemy pour l'écriture vers la table clean (défaut : config.load_postgres_config).",
    )
    parser.add_argument(
        "--clean-table-name",
        type=str,
        default=SETTINGS["postgres"]["table_clean"],
        help="Table Postgres cible pour l'insertion des données nettoyées.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    args = parse_arguments()

    preprocessing_data = prepare_data(
        source=args.source,
        data_path=args.data_path,
        connection_uri=args.connection_uri,
        table_name=args.table_name,
        api_url=args.api_url,
        test_size=args.test_size,
        random_state=args.random_state,
        split_strategy=args.split_strategy,
        missing_threshold=args.missing_threshold,
        drop_high_missing=args.drop_high_missing,
        save_report=not args.no_save_report,
    )

    display_preprocessing_summary(preprocessing_data)
    check_preprocessor_works(preprocessing_data)
