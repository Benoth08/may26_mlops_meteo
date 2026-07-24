"""
Logique d'inférence : rejoue le feature engineering de build_features sur
des données brutes, puis appelle le modèle. Aucun code FastAPI ici — ce
module doit être testable en pur Python (pandas in, dict/array out).
"""
import numpy as np
import pandas as pd
from fastapi import Request
from sqlalchemy import text

from core.logger import get_logger

from constants import (
    TABLE_RAW,
    DB_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
    COLUMN_CONSTRAINTS,
    REQUIRED_COLUMNS,
    FEATURE_COLUMNS,
    TECHNICAL_COLUMNS,
    WIND_DIRECTION_COLUMNS,
    CATEGORICAL_COLUMNS,
    normalize_column_name,
    normalize_data
)
from deps import get_model, get_engine

from build_features import (
    validate_schema, convert_types, encode_rain_today, parse_date_column, add_temporal_features,
    add_cyclical_features, encode_wind_directions, add_weather_features, drop_unused_columns
)

logger = get_logger("inference")

def validate_location(location: str) -> None:
    """
    Vérifie que la station est connue du modèle.
    """
    _, metadata = get_model()

    known_locations = (
        metadata
        .get("location", {})
        .get("known_values", [])
    )

    if not known_locations:
        raise ValueError(
            "Aucune station connue n'est disponible dans les métadonnées du modèle."
        )

    key = location.strip()
    
    if location not in known_locations:
        raise ValueError(
            f"Localisation inconnue : '{location}'. "
            f"{len(known_locations)} stations disponibles."
        )
        
# -------------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------------
def featurize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transforme des données brutes en features, exactement comme build_features."""
    df = normalize_data(df_raw.copy())
    
    try:
        logger.info({
            "event": "inference_input_columns",
            "columns": df.columns.tolist()
        })
        
        # suppression de la cible
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])
        
        # validation station
        if "location" in df.columns:
            for location in df["location"].dropna().unique():
                validate_location(location)
        
        
        # validation schéma attendu par le modèle
        _, metadata = get_model()

        expected_features = (
            metadata["features"]["numeric"]
            +
            metadata["features"]["categorical"]
        )

        validate_schema(df, expected_features) 
        
        # même pipeline que training
        df_conv = convert_types(df)
               
        if not df_conv.empty:
            df_conv = encode_rain_today(df_conv)
            df_conv = parse_date_column(df_conv)   # à conserver ?
            df_conv = add_temporal_features(df_conv)
            df_conv = add_cyclical_features(df_conv)
            df_conv = encode_wind_directions(df_conv)
            df_conv = add_weather_features(df_conv)
            df_conv = drop_unused_columns(df_conv)
            
            return df_conv
        else:
            return pd.DataFrame()
    except Exception as e:
        logger.exception({
            "event": "featurization_failed",
            "error_type": type(e).__name__,
            "error": str(e),
            "input_columns": df.columns.tolist()
        })

        raise


# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------
def proba_of(X: pd.DataFrame) -> np.ndarray:
    """Probabilité de pluie (classe 1) pour chaque ligne."""
    pipeline, _ = get_model()
    if pipeline is None:
        raise RuntimeError("Modèle non chargé")

    return pipeline.predict_proba(X)[:, 1]

# -------------------------------------------------------------------------
# Lecture DB pour démonstration
# -------------------------------------------------------------------------
def load_row_from_db(request: Request, date: str, location: str):
    engine = get_engine(request)
    if engine is None:
        raise RuntimeError("Engine non initialisé")

    validate_location(location)
    
    cols = ", ".join(DB_COLUMNS.keys())
    query = text(
        f"SELECT {cols} FROM {TABLE_RAW} "
        "WHERE date = :d AND location = :loc "
        "ORDER BY {IMPORT_DATE_COLUMN} DESC LIMIT 1"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"d": date, "loc": location})

    if df.empty:
        return None, None
    actual = df[normalize_column_name(TARGET)].iloc[0]
    return df, actual
