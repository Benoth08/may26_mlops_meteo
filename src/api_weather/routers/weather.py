"""
Endpoints de consultation des données météo brutes stockées en base :
    GET /weather/features       liste des paramètres météo autorisés
    GET /weather                1 paramètre météo pour une date et une station
    GET /weather/all-features   toutes les données météo pour une date et une station
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request, HTTPException
from sqlalchemy import text


from core.logger import get_logger

from auth import check_auth
from constants import ALLOWED_FEATURES, TABLE_RAW, IMPORT_DATE_COLUMN, normalize_column_name
from deps import (
    check_model,
    check_database,
    get_engine,
    get_model,
)

logger = get_logger("api")

router = APIRouter(prefix="/weather", tags=["weather"])

@router.get("/features", responses={
    200: {"description": "OK"},
    500: {"description": "Erreur serveur"},
})
def get_weather_features():
    """Retourne la liste des paramètres météo autorisés."""
    try:
        _, metadata = get_model()
        
        features = metadata.get(
            "features",
            []
        )
        
        known_locations = (
            metadata
            .get("location", {})
            .get("known_values", [])
        )            
        
        return {
            "status": "success",
            "features": features,
            "locations_count": len(known_locations),
            "locations": known_locations
        }
    except Exception as e:
        logger.exception({
            "event": "weather_features_failed",
            "error": str(e)
        })

        raise HTTPException(
            status_code=500,
            detail="Erreur interne lors de l'analyse des features."
        )
    
    
@router.get("", responses={
    200: {"description": "OK"},
    400: {"description": "Paramètre incorrect (date ou champ météo)"},
    404: {"description": "Aucun import disponible"},
    500: {"description": "Erreur serveur"},
})
def get_weather(
    request: Request,
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
    location: str = Query(..., description="Nom de la station météo"),
    feature: str = Query("Rainfall", description="Nom de la colonne météo à consulter"),
    auth: bool = Depends(check_auth),
):
    """Retourne une donnée météo pour une date, une localisation et un paramètre donnés."""
    if feature not in ALLOWED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "error": f"Paramètre '{feature}' non disponible",
                "allowed_features": sorted(ALLOWED_FEATURES),
            },
        )

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "error": f"Format de date invalide : '{date}'. Format attendu : YYYY-MM-DD",
            },
        )

    engine = get_engine(request)
    if engine is None:
        raise HTTPException(500, "Engine non initialisé")

    feature_norm = normalize_column_name(feature)
    q = text(
        f"""
        SELECT {feature_norm}
        FROM {TABLE_RAW}
        WHERE date = :d
            AND location = :loc
            AND {IMPORT_DATE_COLUMN} = (
            SELECT MAX({IMPORT_DATE_COLUMN})
            FROM {TABLE_RAW}
        )
        LIMIT 1;
        """
    )
    

    try:
        with engine.connect() as conn:
            row = conn.execute(q, {"d": date, "loc": location}).fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Aucune donnée trouvée pour cette date / station",
        )

    return {
        "status": "success",
        "date": date,
        "location": location,
        "feature": feature,
        "value": row[0],
    }


@router.get("/all-features", responses={
    200: {"description": "OK"},
    400: {"description": "Paramètre météo indisponible"},
    404: {"description": "Aucun import disponible"},
    500: {"description": "Erreur serveur"},
})
def get_weather_all_features(
    request: Request,
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
    location: str = Query(..., description="Nom de la station météo (obligatoire)"),
    auth: bool = Depends(check_auth),
):
    """Retourne tous les paramètres météo pour une date et une station données."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "error": f"Format de date invalide : '{date}'. Format attendu : YYYY-MM-DD",
            },
        )

    allowed_cols_sorted = sorted(ALLOWED_FEATURES)
    allowed_cols_sql = ", ".join(normalize_column_name(c) for c in allowed_cols_sorted)

    q = text(
        f"""
        SELECT {allowed_cols_sql}
        FROM {TABLE_RAW}
        WHERE date = :d
            AND location = :loc
            AND {IMPORT_DATE_COLUMN} = (
            SELECT MAX({IMPORT_DATE_COLUMN})
            FROM {TABLE_RAW}
        )
        LIMIT 1;
        """
    )
    engine = get_engine(request)
    if engine is None:
        raise HTTPException(500, "Engine non initialisé")

    try:
        with engine.connect() as conn:
            row = conn.execute(q, {"d": date, "loc": location}).fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnée trouvée pour {location} à la date {date}",
        )

    data = dict(zip(allowed_cols_sorted, row))

    return {
        "status": "success",
        "location": location,
        "date": date,
        "data": data,
    }
