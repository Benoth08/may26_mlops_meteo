"""
Endpoints d'inférence :
    POST /predict            1 observation (JSON)          -> prédiction
    POST /predict-batch      fichier CSV de nouvelles obs.  -> prédictions
    GET  /predict-from-db    rejoue une ligne déjà ingérée  -> test / démo
"""
import time
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from core.logger import get_logger

from auth import check_auth
from constants import MAX_BATCH_ROWS, PREDICTION_THRESHOLD, normalize_data
from inference import featurize, load_row_from_db, proba_of
from schemas import WeatherInput
from prometheus_metrics import (
    record_prediction,
    record_batch,
)
from deps import ModelUnavailableError

router = APIRouter(prefix="/predict", tags=["predict"])

logger = get_logger("predict")


@router.post("/")
def predict(item: WeatherInput, auth: bool = Depends(check_auth)):
    """Prédiction sur UNE nouvelle observation envoyée en JSON."""
    try:
        X = featurize(pd.DataFrame([item.model_dump()]))
        if X is None or X.empty:
            logger.warning({
                "event": "prediction_rejected",
                "reason": "empty_features"
            })

            raise HTTPException(status_code=400, detail="Item invalide : aucune feature générée après prétraitement.")
        logger.info({
            "event": "prediction_features_ready",
            "columns": X.columns.tolist(),
            "shape": X.shape
        })
        start = time.perf_counter()
        p = float(proba_of(X)[0])
        duration = time.perf_counter() - start

        record_prediction(probability=p, duration=duration)

        return {"rain_tomorrow": "Yes" if p >= PREDICTION_THRESHOLD else "No", "probability": round(p, 4)}

    except (HTTPException, ModelUnavailableError):
        raise
    except ValueError as e:
        # Erreurs métier : localisation inconnue,
        # feature invalide, données incohérentes...
        logger.warning({
            "event": "prediction_validation_failed",
            "error": str(e)
        })

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.exception({
            "event": "prediction_failed",
            "error": str(e)
        })

        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de la prédiction."
        )

@router.post("/batch")
def predict_batch(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
    """Prédiction par lot : envoie un CSV de nouvelles observations (mêmes colonnes
    que weatherAUS), reçois une prédiction par ligne."""
    try:
        raw = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(400, f"CSV illisible : {e}")

    # Garde-fou anti-DoS mémoire.
    if len(raw) > MAX_BATCH_ROWS:
        raise HTTPException(
            400,
            f"CSV trop volumineux : {len(raw)} lignes (max {MAX_BATCH_ROWS})."
        )
    
    try:
        X = featurize(raw)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"Données invalides : {e}")
    
    if X.empty:
        raise HTTPException(400, "Aucune ligne exploitable (dates invalides ?).")

    start = time.perf_counter()
    p = proba_of(X)
    duration = time.perf_counter() - start

    record_batch(probabilities=p, duration=duration)

    # featurize() normalise les noms de colonnes : on fait de même pour
    # retrouver date / station, que le CSV soit en "Date" ou en "date".
    raw_norm = normalize_data(raw.copy())

    out = pd.DataFrame({
        "Date": raw_norm.loc[X.index, "date"].values,
        "Location": raw_norm.loc[X.index, "location"].values,
        "rain_tomorrow": np.where(p >= PREDICTION_THRESHOLD, "Yes", "No"),
        "probability": p.round(4),
    })
    return {"count": int(len(out)), "predictions": out.to_dict(orient="records")}


@router.get("/from-db")
def predict_from_db(
    request: Request,
    date: str,
    location: str,
    auth: bool = Depends(check_auth),
):
    """Rejoue une ligne DÉJÀ ingérée (test/démo). ⚠️ Non pertinent si la date
    faisait partie de l'entraînement : viser une date hors période de train."""
    df, actual = load_row_from_db(request, date, location)
    if df is None:
        raise HTTPException(404, f"Aucune donnée pour {location} le {date}")
    
    try:
        X = featurize(df)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"Données invalides : {e}")
        
    if X.empty:
        raise HTTPException(400, "Ligne écartée au prétraitement (date invalide).")
    start = time.perf_counter()
    p = float(proba_of(X)[0])
    duration = time.perf_counter() - start
    record_prediction(probability=p, duration=duration)
    return {
        "date": date,
        "location": location,
        "rain_tomorrow_pred": "Yes" if p >= PREDICTION_THRESHOLD else "No",
        "probability": round(p, 4),
        "rain_tomorrow_actual": actual,
    }
