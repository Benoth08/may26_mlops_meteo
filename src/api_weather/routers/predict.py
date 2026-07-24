"""
Endpoints d'inférence :
    POST /predict            1 observation (JSON)          -> prédiction
    POST /predict-batch      fichier CSV de nouvelles obs.  -> prédictions
    GET  /predict-from-db    rejoue une ligne déjà ingérée  -> test / démo
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from core.logger import get_logger

from auth import check_auth
from constants import MAX_BATCH_ROWS
from inference import featurize, load_row_from_db, proba_of
from schemas import WeatherInput

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
        p = float(proba_of(X)[0])
        return {"rain_tomorrow": "Yes" if p >= 0.5 else "No", "probability": round(p, 4)}
    except HTTPException:
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
async def predict_batch(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
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
