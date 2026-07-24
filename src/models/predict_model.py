#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Prédiction

    Description :
        Chargement du modèle et prédiction de la pluie à partir de données brutes

    Version :
        1.0.0

    Historique :
        2026-07-10  -  Création du module
===============================================================================
"""

import os
from core.logger import get_logger
from core.settings import SETTINGS
for _env_key, _env_val in SETTINGS["threads"].items():
    os.environ.setdefault(_env_key, str(_env_val))
from core.metadata import NUMERIC_COLUMNS, TARGET, TECHNICAL_COLUMNS, normalize_column_name, normalize_data

from build_features import (
    encode_rain_today, parse_date_column, add_temporal_features,
    add_cyclical_features, encode_wind_directions, add_weather_features, drop_unused_columns
)



from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from pathlib import Path

import joblib
import pandas as pd

logger = get_logger("predict_model")


MODELS_DIR = Path(SETTINGS["paths"]["models"])
MODEL_PATH = (MODELS_DIR / SETTINGS["models"]["model"])
MODEL_PKL_PATH = (MODELS_DIR / SETTINGS["models"]["model_pkl"])

def load_model(model_path=None):
    if model_path is None:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            model_path = Path(MODEL_PKL_PATH)
    return joblib.load(model_path)


def featurize(df_raw):
    df = df_raw.copy()
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = encode_rain_today(df)
    df = parse_date_column(df)
    df = add_temporal_features(df)
    df = add_cyclical_features(df)
    df = encode_wind_directions(df)
    df = add_weather_features(df)
    df = drop_unused_columns(df)
    drop = [TARGET] + TECHNICAL_COLUMNS
    return df.drop(columns=[c for c in drop if c in df.columns])


def predict(df_raw, model=None):
    if model is None:
        model = load_model()
    X = featurize(df_raw)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba
