"""
===============================================================================
    WeatherAUS - Métadonnées du schéma
    ---------------------------------------------------------------------------
    Ce module construit toutes les métadonnées dérivées de
    SETTINGS["constraints"].

    Il constitue l'unique point d'accès aux listes de colonnes utilisées
    dans le projet.
===============================================================================
"""

from .settings import SETTINGS
from .helpers_dataframe import (
    get_all_columns,
    get_feature_columns,
    get_numeric_columns,
    get_columns_by_metadata,
    normalize_column,
)

import pandas as pd


# ============================================================================
# Contraintes de schéma
# ============================================================================

COLUMN_CONSTRAINTS = SETTINGS["constraints"]

def build_feature_constraints(constraints: dict) -> dict:
    """
    Extrait uniquement les colonnes 'features' depuis SETTINGS["constraints"].

    Retourne un dict :
    {
        "MinTemp": {...},
        "MaxTemp": {...},
        ...
    }
    """
    return {
        col: cfg
        for col, cfg in constraints.items()
        if cfg.get("feature", False)
    }

FEATURE_CONSTRAINTS = build_feature_constraints(COLUMN_CONSTRAINTS)

# ============================================================================
# Colonnes métier
# ============================================================================

FEATURE_COLUMNS = tuple(get_feature_columns(COLUMN_CONSTRAINTS))
ALLOWED_FEATURES = frozenset(FEATURE_COLUMNS)

TARGET = SETTINGS["target"]["column_norm"]

# ============================================================================
# Colonnes techniques
# ============================================================================

TECHNICAL_COLUMNS = get_columns_by_metadata(COLUMN_CONSTRAINTS,"technical")
NON_TECHNICAL_COLUMNS = get_columns_by_metadata(COLUMN_CONSTRAINTS,"technical", False)

# ============================================================================
# Colonnes numériques
# ============================================================================

NUMERIC_COLUMNS = get_numeric_columns(COLUMN_CONSTRAINTS)

# ============================================================================
# Colonnes catégorielles
# ============================================================================

CATEGORICAL_COLUMNS = get_columns_by_metadata(
    COLUMN_CONSTRAINTS,
    {
        "feature": True,
        "type": "string"
    }
)

# ============================================================================
# Colonnes de direction du vent
# ============================================================================

WIND_DIRECTION_COLUMNS = get_columns_by_metadata(COLUMN_CONSTRAINTS, "wind_direction")

# ============================================================================
# Colonnes obligatoires
# ============================================================================

REQUIRED_COLUMNS = get_columns_by_metadata(COLUMN_CONSTRAINTS, "required")

# ============================================================================
# Colonnes PostgreSQL normalisées
# ============================================================================

def get_db_columns(column_constraints: dict) -> dict[str, str]:
    """
    Construit le mapping des noms de colonnes PostgreSQL vers les noms
    métier utilisés par le projet.

    Exemple
    -------
    {
        "min_temp": "MinTemp",
        "max_temp": "MaxTemp",
        "rain_tomorrow": "RainTomorrow",
        ...
    }
    """
    return {
        metadata["norm_name"]: column
        for column, metadata in column_constraints.items()
    }
    
DB_COLUMNS = get_db_columns(COLUMN_CONSTRAINTS)


def get_normalized_columns(column_constraints: dict) -> dict[str, str]:
    """
    Construit le mapping des noms métier vers les noms normalisés.

    Exemple
    -------
    {
        "MinTemp": "min_temp",
        "MaxTemp": "max_temp",
        ...
    }
    """
    return {
        column: metadata["norm_name"]
        for column, metadata in column_constraints.items()
    }

NORMALIZED_COLUMNS = get_normalized_columns(COLUMN_CONSTRAINTS)


def normalize_column_name(
    col: str
) -> str:
    """
    Normalise un nom de colonne.

    Si un mapping `norm_name` est défini dans SETTINGS, il est utilisé.
    Sinon applique les transformations 
    """
    if col is None:
        raise ValueError("Column name cannot be None")
        
    # Utilisation du nom normalisé défini dans la configuration
    metadata = COLUMN_CONSTRAINTS.get(col, {})

    if "norm_name" in metadata:
        return metadata["norm_name"]
        
    # Suppression accents
    return normalize_column(col)


def normalize_data(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalise les colonnes du DataFrame pour éviter les erreurs de casse.
    """
    # Normalisation des colonnes du DF
    df.columns = [normalize_column(c) for c in df.columns]

    return df
