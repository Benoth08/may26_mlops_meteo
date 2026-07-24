"""
Dépendances FastAPI partagées entre routers : authentification, accès DB,
chargement du modèle. Regroupées ici pour être injectées via Depends(...)
dans n'importe quel router, sans dupliquer la logique.

État partagé : DB_CONFIG et API_CREDENTIALS ne sont plus des globals de
module (impossible à maintenir proprement une fois le code éclaté en
plusieurs fichiers) mais vivent dans `request.app.state`, rempli une seule
fois au startup (cf. api/main.py).
"""

import joblib
import time

from pathlib import Path
from threading import Lock

from fastapi import Request
from sqlalchemy import create_engine, text

from core.logger import get_logger
from constants import MODEL_PATH, TABLE_RAW, IMPORT_DATE_COLUMN

logger = get_logger("api")


# ── Modèle ───────────────────────────────────────────────────────────────
class ModelManager:
    """
    Gestion thread-safe du modèle.

    - Recharge automatique lorsque le fichier change.
    - Chargement du modèle hors verrou.
    - Remplacement atomique des références.
    """

    def __init__(self):
        self.model = None
        self.metadata = None
        self.version = None

        self._lock = Lock()

    def _get_version(self):
        """
        Retourne la version du modèle basée sur la date
        de modification du fichier.
        """
        if not MODEL_PATH.exists():
            return None

        return MODEL_PATH.stat().st_mtime



    def _load_artifact(self):
        """
        Charge le modèle depuis le disque.
        """
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle absent : {MODEL_PATH}"
            )


        start = time.perf_counter()
        artifact = joblib.load(MODEL_PATH)
        duration = time.perf_counter() - start
        logger.info(
            {
                "event": "model_load_duration",
                "seconds": round(duration, 3),
                "path": str(MODEL_PATH),
            }
        )
        
        if isinstance(artifact, dict):
            model = artifact.get("pipeline")
            metadata = artifact.get("metadata", {})
        else:
            model = artifact
            metadata = {}

        if model is None:
            raise RuntimeError(
                "Artefact chargé mais aucun modèle trouvé"
            )

        return (
            model,
            metadata,
            self._get_version()
        )
    
    def get_model(self):
        """
        Retourne le modèle courant.

        Recharge automatiquement si le fichier a changé.
        """
        
        logger.info({
            "event": "check_model",
            "model_path": str(MODEL_PATH),
            "exists": MODEL_PATH.exists(),
            "version": self.version,
            "current_version": self._get_version()
        })

        current_version = self._get_version()

        # Le fichier modèle n'existe plus
        if current_version is None:
            with self._lock:
                self.model = None
                self.metadata = None
                self.version = None

            raise FileNotFoundError(
                f"Modèle absent : {MODEL_PATH}"
            )


        # Modèle déjà chargé et inchangé
        if (
            current_version == self.version
            and self.model is not None
        ):
            return self.model, self.metadata


        with self._lock:

            current_version = self._get_version()

            if current_version is None:
                self.model = None
                self.metadata = None
                self.version = None

                raise FileNotFoundError(
                    f"Modèle absent : {MODEL_PATH}"
                )

            if (
                current_version == self.version
                and self.model is not None
            ):
                return self.model, self.metadata


        # Chargement hors lock
        model, metadata, loaded_version = self._load_artifact()


        with self._lock:
            self.model = model
            self.metadata = metadata
            self.version = loaded_version


        logger.info(
            {
                "event": "model_loaded",
                "version": loaded_version,
            }
        )

        return self.model, self.metadata
    

    def current_version(self):
        return self.version 
        

model_manager = ModelManager()


def get_model():
    pipeline, metadata = model_manager.get_model()

    if pipeline is None:
        raise RuntimeError("Le modèle n'a pas pu être chargé.")

    return pipeline, metadata


def check_model():
    """
    """
    try:
        model, metadata = model_manager.get_model()

        result = {
            "status": "loaded",
            "available": model is not None,
        }

        if metadata:
            result["model_version"] = metadata.get(
                "model_version"
            )

        return result


    except FileNotFoundError:
        return {
            "status": "missing",
            "available": False,
        }


    except Exception as e:
        logger.exception(
            "Erreur lors du chargement du modèle"
        )

        return {
            "status": "error",
            "available": False,
            "error": str(e),
        }
    
    
# ── Base de données ─────────────────────────────────────────────────────
def get_engine(request: Request):
    """Crée un engine Postgres à la demande à partir de la config chargée au startup."""
    cfg = request.app.state.db_config
    if not cfg:
        return None
    try:
        return create_engine(cfg.sqlalchemy_uri)
    except Exception as e:
        logger.error({"event": "engine_creation_failed", "error": str(e)}, exc_info=True)
        return None


def check_database(request: Request):
    engine = get_engine(request)
    if engine is None:
        return {"status": "unavailable", "error": "Base de données inaccessible"}

    try:
        q = text(f"SELECT MAX({IMPORT_DATE_COLUMN}) FROM {TABLE_RAW};")
        with engine.connect() as conn:
            row = conn.execute(q).fetchone()
        return {
            "status": "connected",
            "last_import": row[0].isoformat() if row and row[0] else "Aucun import disponible",
        }
    except Exception as e:
        logger.error({"event": "check_database_failed", "error": str(e)}, exc_info=True)
        return {"status": "error", "error": "Schéma des données invalide"}

