#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Weather Integration
    
    Description :
        Integration des données météo en base via fonctions WeatherDataLoader
        
    Version :
        1.0.0

    Historique :
        2026-06-11  -  Création du module
===============================================================================
"""

import sys
import json
import time

from core.logger import get_logger
from core.settings import SETTINGS
from weather_loader import WeatherDataLoader
from core.config import load_postgres_config, PostgresConfig, ConfigError

logger = get_logger("entrypoint", console=False)
 
CSV_PATH = str(SETTINGS["paths"]["data"] / SETTINGS["models"]["rawdata"])


def main():

    # -----------------------------
    # 1. Charger la configuration Postgres
    # -----------------------------
    try:
        cfg = load_postgres_config()
    except ConfigError as e:
        logger.error({"event": "config_error", "error": str(e)})
        print(json.dumps({"status": "failed", "error": str(e)}))
        sys.exit(1)

    # -----------------------------
    # 2. Lancer l’integration
    # -----------------------------
    try:
        
        start = time.time()

        logger.info({"event": "integration_start", "csv_path": CSV_PATH, "host": cfg.host})
        
        loader = WeatherDataLoader(cfg)
        
        result = loader.load_csv(CSV_PATH)
        duration = time.time() - start

        # -----------------------------
        # 3. Retourner XCom → JSON
        # -----------------------------

        # 3.1 Vérifier que result est un dict
        if not isinstance(result, dict):
            raise Exception("Résultat d'import invalide : attendu un dict")

        # 2. Vérifier la présence du champ status
        if "status" not in result:
            raise Exception("Champ 'status' manquant dans le résultat d'import")

        status = result["status"]

        # 3. Cas : status = failed
        if status == "failed":
            # Test : champ error
            if "error" not in result:
                raise Exception("Import échoué mais champ 'error' manquant")
            
            # Message d’erreur complet
            raise Exception(
                f"Import FAILED : {result['error']}"
            )

        # -----------------------------
        # 4. Cas : status = success
        # -----------------------------
        if status == "success":

            # Test : rows_imported
            if "rows_imported" not in result:
                raise Exception("Import réussi mais champ 'rows_imported' manquant")

            # Test : archive_path
            if "archive_path" not in result:
                raise Exception("Import réussi mais champ 'archive_path' manquant")

            logger.info({
                "event": "integration_success",
                "rows_imported": result["rows_imported"],
                "archive_path": result["archive_path"],
                "duration_seconds": round(duration, 2),
            })
            
            # Log succès
            print(json.dumps({
                "status": "success",
                "rows_imported": result["rows_imported"],
                "archive_path": str(result["archive_path"])
            }))

            return result

        # -----------------------------
        # 5. Cas : status inconnu
        # -----------------------------
        raise Exception(f"Valeur de status inconnue : {status}")

    except Exception as e:
        logger.error({"event": "integration_failed", "error": str(e)}, exc_info=True)
        print(json.dumps({"status": "failed", "error": str(e)}))
        raise


if __name__ == "__main__":
    main()
    
