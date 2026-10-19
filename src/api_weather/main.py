#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Weather API / Predict next-day rain in Australia

    Description :
        Point d'entrée de l'API REST. Ne contient QUE le bootstrap
        (création de l'app FastAPI, chargement config au startup, montage
        des routers) — toute la logique métier vit dans inference.py/deps.py
        et les endpoints dans routers/.

        - GET  /                      infos + lien vers la doc Swagger
        - GET  /health                état de l'API + modèle chargé
        - GET  /last-import           infos sur le dernier import de données
        - GET  /weather/features      liste des paramètres météo autorisés
        - GET  /weather               1 paramètre météo pour une date et une station
        - GET  /weather/all-features  toutes les données météo pour une date et une station
        - POST /predict                1 observation (JSON)             -> prédiction
        - POST /predict-batch          fichier CSV de nouvelles obs.    -> prédictions
        - GET  /predict-from-db        rejoue une ligne déjà ingérée    -> test / démo

    Version :
        2.1.0

    Historique :
        2026-06-11  -  Création du module
        2026-06-26  -  Ajout des endpoints de prévision
        2026-07-09  -  Découpage en package (constants/schemas/deps/inference/routers)
===============================================================================
"""
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy import create_engine

from core.logger import get_logger
from core.config import ConfigError, load_postgres_config
from constants import MODEL_PATH, API_VERSION, THREADS_SETTINGS
from deps import model_manager, ModelUnavailableError

# Doit être posé AVANT l'import de numpy/sklearn
for _env_key, _env_val in THREADS_SETTINGS.items():
    os.environ.setdefault(_env_key, str(_env_val))


from routers import meta, predict, weather
from prometheus_metrics import (
    setup_metrics,
    update_model_status,
    load_model_evaluation_metrics,
)

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ============================================================
    # STARTUP
    # ============================================================
    # Config postgres
    try:
        app.state.db_config = load_postgres_config()
    except ConfigError as e:
        logger.error({
            "event": "startup_failed",
            "error": str(e)
        })
        raise RuntimeError(
            f"Configuration PostgreSQL invalide : {e}"
        )

    # Un seul engine (et donc un seul pool de connexions) pour toute l'app.
    app.state.db_engine = create_engine(
        app.state.db_config.sqlalchemy_uri,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=3,
        # SQL_ECHO=1 : SQLAlchemy logue chaque requête exécutée et ses paramètres
        echo=os.environ.get("SQL_ECHO") == "1",
    )

    # Authentification API
    expected_username = os.environ.get("API_AUTH_USERNAME", "")
    expected_password_hash = os.environ.get("API_AUTH_PASSWORD_HASH", "")

    if not expected_username or not expected_password_hash:
        logger.error({
            "event": "startup_failed",
            "error": "API_AUTH_USERNAME/API_AUTH_PASSWORD_HASH manquants"
        })
        raise RuntimeError(
            "API_AUTH_USERNAME et API_AUTH_PASSWORD_HASH "
            "doivent être définis dans l'environnement."
        )

    app.state.api_credentials = {
        "user": expected_username,
        "password": expected_password_hash,
    }

    # Chargement anticipé du modèle
    try:
        _, metadata = model_manager.get_model()

        update_model_status(
            loaded=True,
            metadata=metadata,
        )

        load_model_evaluation_metrics()

        logger.info(
            {
                "event": "startup_ready",
                "model": "loaded"
            }
        )
    except FileNotFoundError as e:
        update_model_status(
            loaded=False,
        )

        logger.warning(
            {
                "event": "model_unavailable_at_startup",
                "error": str(e),
            }
        )
        
    logger.info({
        "event": "api_started",
        "model_path": str(MODEL_PATH),
        "host": app.state.db_config.host,
        "port": app.state.db_config.port,
    })

    # ============================================================
    # APPLICATION ACTIVE
    # ============================================================
    yield

    # ============================================================
    # SHUTDOWN
    # ============================================================
    app.state.db_engine.dispose()
    logger.info({
        "event": "api_shutdown"
    })


app = FastAPI(
    title="Weather Inference API",
    description=(
        "Prédit s'il pleuvra demain (RainTomorrow) à partir d'observations "
        "météo brutes. Sécurisée par authentification basique (identifiants "
        "fournis séparément, cf. équipe MLOps)."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# Initialisés à vide pour que deps.py puisse lire app.state sans AttributeError
# même avant que le startup ait tourné (ex: tests qui montent l'app sans TestClient
# lifespan). Remplis pour de bon dans startup_event().
app.state.db_config = {}
app.state.db_engine = None
app.state.api_credentials = {}

@app.exception_handler(ModelUnavailableError)
def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
    """Modèle absent : 503 explicite plutôt qu'un 500 générique."""
    logger.warning({
        "event": "model_unavailable",
        "path": request.url.path,
        "error": str(exc),
    })
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Aucun modèle n'est disponible pour le moment : le modèle "
                "n'a pas encore été entraîné ou promu. Réessayez plus tard."
            )
        },
    )

app.include_router(meta.router)
app.include_router(weather.router)
app.include_router(predict.router)

# Monitoring Prometheus
setup_metrics(app)

