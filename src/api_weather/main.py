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

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from core.logger import get_logger
from core.config import ConfigError, PostgresConfig, load_postgres_config
from constants import MODEL_PATH, API_VERSION, THREADS_SETTINGS
from deps import model_manager

# Doit être posé AVANT l'import de numpy/sklearn
for _env_key, _env_val in THREADS_SETTINGS.items():
    os.environ.setdefault(_env_key, str(_env_val))


from routers import meta, predict, weather

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
        model_manager.get_model()
        logger.info(
            {
                "event": "startup_ready",
                "model": "loaded"
            }
        )
    except FileNotFoundError as e:
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
app.state.api_credentials = {}

app.include_router(meta.router)
app.include_router(weather.router)
app.include_router(predict.router)
