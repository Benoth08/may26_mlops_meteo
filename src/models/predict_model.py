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

from sklearn.experimental import enable_iterative_imputer

from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import (
    encode_rain_today, parse_date_column, add_temporal_features,
    add_cyclical_features, encode_wind_directions, add_weather_features,
    drop_unused_columns, TARGET, TECHNICAL_COLUMNS,
)

RAW_NUMERIC = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]


def load_model(model_path=None):
    if model_path is None:
        model_path = Path("models/model.joblib")
        if not model_path.exists():
            model_path = Path("models/model.pkl")
    return joblib.load(model_path)


def featurize(df_raw):
    df = df_raw.copy()
    for c in RAW_NUMERIC:
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
