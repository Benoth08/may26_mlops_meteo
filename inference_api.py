"""API d'inférence Weather — prédit RainTomorrow sur de NOUVELLES données.

Endpoints
---------
GET  /                 infos + lien vers la doc Swagger
GET  /health           état de l'API + modèle disponible
POST /predict          1 observation (JSON)              -> prédiction
POST /predict-batch    fichier CSV de nouvelles obs.     -> prédictions
GET  /predict-from-db  rejoue une ligne déjà ingérée     -> test / démo
"""
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (dépickler IterativeImputer)

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from src.features.build_features import (
    encode_rain_today, parse_date_column, add_temporal_features,
    add_cyclical_features, encode_wind_directions, add_weather_features,
    drop_unused_columns, TARGET, TECHNICAL_COLUMNS,
)

# Constantes
RAW_NUMERIC = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]

# colonnes Postgres (minuscules) -> noms attendus par build_features (PascalCase)
DB_COLUMNS = {
    "date": "Date", "location": "Location", "mintemp": "MinTemp", "maxtemp": "MaxTemp",
    "rainfall": "Rainfall", "evaporation": "Evaporation", "sunshine": "Sunshine",
    "windgustdir": "WindGustDir", "windgustspeed": "WindGustSpeed",
    "winddir9am": "WindDir9am", "winddir3pm": "WindDir3pm",
    "windspeed9am": "WindSpeed9am", "windspeed3pm": "WindSpeed3pm",
    "humidity9am": "Humidity9am", "humidity3pm": "Humidity3pm",
    "pressure9am": "Pressure9am", "pressure3pm": "Pressure3pm",
    "cloud9am": "Cloud9am", "cloud3pm": "Cloud3pm",
    "temp9am": "Temp9am", "temp3pm": "Temp3pm",
    "raintoday": "RainToday", "raintomorrow": "RainTomorrow",
}

# Chargement du modèle (pipeline complet : prétraitement + LightGBM)
# Chargement tolérant : l'API demarre meme sans modele. Le modele est charge
# a la premiere prediction, une fois qu'il existe dans le dossier models.
_model = None


def get_model():
    global _model
    if _model is None:
        path = Path("models/model.joblib")
        if not path.exists():
            path = Path("models/model.pkl")
        if not path.exists():
            raise HTTPException(
                503, "Aucun modèle disponible. Lancez d'abord l'entraînement.")
        _model = joblib.load(path)
    return _model

app = FastAPI(
    title="Weather Inference API",
    description="Prédit s'il pleuvra demain (RainTomorrow) à partir d'observations météo brutes.",
    version="2.0.0",
)


# Schéma d'entrée (1 observation)
class WeatherInput(BaseModel):
    Date: str
    Location: str
    MinTemp: Optional[float] = None
    MaxTemp: Optional[float] = None
    Rainfall: Optional[float] = None
    Evaporation: Optional[float] = None
    Sunshine: Optional[float] = None
    WindGustDir: Optional[str] = None
    WindGustSpeed: Optional[float] = None
    WindDir9am: Optional[str] = None
    WindDir3pm: Optional[str] = None
    WindSpeed9am: Optional[float] = None
    WindSpeed3pm: Optional[float] = None
    Humidity9am: Optional[float] = None
    Humidity3pm: Optional[float] = None
    Pressure9am: Optional[float] = None
    Pressure3pm: Optional[float] = None
    Cloud9am: Optional[float] = None
    Cloud3pm: Optional[float] = None
    Temp9am: Optional[float] = None
    Temp3pm: Optional[float] = None
    RainToday: Optional[str] = None


# Prétraitement (rejoue le feature engineering de build_features)
def featurize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Transforme des données brutes en features, exactement comme build_features."""
    df = df_raw.copy()
    for c in RAW_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = encode_rain_today(df)
    df = parse_date_column(df)        # supprime les lignes à date invalide
    df = add_temporal_features(df)
    df = add_cyclical_features(df)
    df = encode_wind_directions(df)
    df = add_weather_features(df)
    df = drop_unused_columns(df)
    drop = [TARGET] + TECHNICAL_COLUMNS
    return df.drop(columns=[c for c in drop if c in df.columns])


def proba_of(X: pd.DataFrame) -> np.ndarray:
    """Probabilité de pluie (classe 1) pour chaque ligne."""
    return get_model().predict_proba(X)[:, 1]


# Accès base Postgres (pour le rejeu)
def get_engine():
    user = os.getenv("POSTGRES_WTH_USER", "weather")
    pwd = os.getenv("POSTGRES_WTH_PASSWORD", "MLops26")
    db = os.getenv("POSTGRES_WTH_DB", "weather")
    host = os.getenv("POSTGRES_WTH_HOST", "localhost")
    port = os.getenv("POSTGRES_WTH_PORT", "5432")
    return create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}")


def load_row_from_db(date: str, location: str):
    cols = ", ".join(DB_COLUMNS.keys())
    query = text(
        f"SELECT {cols} FROM weather_data_raw "
        "WHERE date = :d AND location = :loc "
        "ORDER BY import_date DESC LIMIT 1"
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn, params={"d": date, "loc": location})
    if df.empty:
        return None, None
    actual = df["raintomorrow"].iloc[0]
    return df.rename(columns=DB_COLUMNS), actual


# Endpoints
@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    path = Path("models/model.joblib")
    if not path.exists():
        path = Path("models/model.pkl")
    return {"status": "ok", "model_available": path.exists()}


@app.post("/predict")
def predict(item: WeatherInput):
    """Prédiction sur UNE nouvelle observation envoyée en JSON."""
    X = featurize(pd.DataFrame([item.model_dump()]))
    if X.empty:
        raise HTTPException(400, "Date invalide : la ligne a été écartée au prétraitement.")
    p = float(proba_of(X)[0])
    return {"rain_tomorrow": "Yes" if p >= 0.5 else "No", "probability": round(p, 4)}


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """Prédiction par lot : envoie un CSV de nouvelles observations (mêmes colonnes
    que weatherAUS), reçois une prédiction par ligne."""
    try:
        raw = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(400, f"CSV illisible : {e}")

    try:
        X = featurize(raw)
    except KeyError as e:
        raise HTTPException(400, f"Colonne manquante dans le CSV : {e}")

    if X.empty:
        raise HTTPException(400, "Aucune ligne exploitable (dates invalides ?).")

    p = proba_of(X)
    out = pd.DataFrame({
        "Date": raw.loc[X.index, "Date"].values,
        "Location": raw.loc[X.index, "Location"].values,
        "rain_tomorrow": np.where(p >= 0.5, "Yes", "No"),
        "probability": p.round(4),
    })
    return {"count": int(len(out)), "predictions": out.to_dict(orient="records")}


@app.get("/predict-from-db")
def predict_from_db(date: str, location: str):
    """Rejoue une ligne DÉJÀ ingérée (test/démo). Non pertinent si la date
    faisait partie de l'entraînement : viser une date hors période de train."""
    df, actual = load_row_from_db(date, location)
    if df is None:
        raise HTTPException(404, f"Aucune donnée pour {location} le {date}")
    X = featurize(df)
    if X.empty:
        raise HTTPException(400, "Ligne écartée au prétraitement (date invalide).")
    p = float(proba_of(X)[0])
    return {
        "date": date,
        "location": location,
        "rain_tomorrow_pred": "Yes" if p >= 0.5 else "No",
        "probability": round(p, 4),
        "rain_tomorrow_actual": actual,
    }
