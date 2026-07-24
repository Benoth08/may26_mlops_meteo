"""
Endpoints de supervision / méta-information sur l'API elle-même :
    GET /             infos + redirection vers la doc Swagger
    GET /health       état de l'API + modèle chargé
    GET /last-import  infos sur le dernier import de données
"""
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from core.logger import get_logger

from auth import check_auth
from constants import API_VERSION, API_NAME, TABLE_RAW, IMPORT_DATE_COLUMN, RUNID_COLUMN, SOURCE_COLUMN
from deps import check_model, check_database, get_engine

logger = get_logger("api")

router = APIRouter(tags=["monitoring"])

@router.get("/")
def root():
    """Redirige vers la documentation Swagger."""
    return RedirectResponse(url="/docs")


@router.get("/health", responses={200: {"description": "OK"}})
def health(request: Request):
    """Vérifie que l'API est fonctionnelle."""
    model_status = check_model()
    db_status = check_database(request)
    
    degraded = (
        model_status["status"] != "loaded"
        or db_status["status"] != "connected"
        or db_status.get("last_import") is None
        or db_status.get("last_import") == "Aucun import disponible"
    )

    return {
        "status": "ok",
        "service": API_NAME,
        "version": API_VERSION,
        "model": model_status,
        "database": db_status,
        "degraded_mode": degraded,
    }


@router.get("/last-import", responses={
    200: {"description": "OK"},
    404: {"description": "Aucun import disponible"},
    500: {"description": "Erreur serveur"},
})
def get_last_import(request: Request, auth: bool = Depends(check_auth)):
    """
    Retourne les informations du dernier import :
    date d'import, fichier source, nombre de lignes importées.
    """
    engine = get_engine(request)
    if engine is None:
        raise HTTPException(status_code=500, detail="Base de données inaccessible")

    q = text(
        f"""
        SELECT
            {IMPORT_DATE_COLUMN},
            {SOURCE_COLUMN},
            COUNT(*) AS rows_imported
        FROM {TABLE_RAW}
        WHERE {IMPORT_DATE_COLUMN} = (
            SELECT MAX({IMPORT_DATE_COLUMN}) FROM {TABLE_RAW}
        )
        GROUP BY {IMPORT_DATE_COLUMN}, {SOURCE_COLUMN}
        ORDER BY {IMPORT_DATE_COLUMN} DESC
        LIMIT 1;
        """
    )

    try:
        with engine.connect() as conn:
            row = conn.execute(q).fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Schéma des données invalide")

    if not row:
        raise HTTPException(
            status_code=404,
            detail={"status": "failed", "error": "Aucun import trouvé"},
        )

    import_date, source_file, rows_imported = row

    return {
        "status": "success",
        "import_date": import_date.isoformat(),
        "source_file": source_file,
        "rows_imported": rows_imported,
    }

@router.get(
    "/model",
    responses={
        200: {"description": "Métadonnées du modèle"},
        500: {"description": "Modèle indisponible ou erreur de chargement"},
    },
)
def get_model_metadata():
    """
    Retourne les métadonnées du modèle actuellement utilisé par l'API.
    """
    try:
        _, metadata = get_model()
        if metadata is None:
            logger.error({"event": "model_metadata_missing"})
            raise HTTPException(
                status_code=500,
                detail="Les métadonnées du modèle sont indisponibles.",
            )

        return {
            "status": "success",
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(
            {
                "event": "get_model_metadata_failed",
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Impossible de charger les métadonnées du modèle.",
        )
