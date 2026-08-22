#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Weather Integration

    Description :
        Integration des données météo en base via WeatherDataLoader

    Version :
        1.0.0

    Historique :
        2026-06-11  -  Création du module
===============================================================================
"""

import sys
import json
import time
import argparse

from pathlib import Path

from core.logger import get_logger
from core.settings import SETTINGS
from core.config import load_postgres_config, ConfigError

from weather_loader import WeatherDataLoader

logger = get_logger("entrypoint", console=False)                                                                                       


def parse_args():
    """
    Récupère le nom du fichier CSV transmis par Airflow.
    """
    parser = argparse.ArgumentParser(
        description="Intégration d'un fichier CSV météo dans PostgreSQL"
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Nom du fichier CSV à intégrer",
    )

    return parser.parse_args()


def resolve_csv_path(filename: str) -> Path:
    """
    Reconstruit le chemin complet du CSV dans le conteneur.

    Le fichier peut être transmis sous l'une des formes :

        weather_batch_001.csv
        weather_batch_001.csv.processing

    Dans le second cas, le suffixe .processing est retiré avant
    l'intégration.
    
    Exemple :
        --file weather_batch_001.csv

    devient :
        /data/weather_batch_001.csv
    """                                       

    # Répertoire DATA dans le conteneur
    data_dir = Path(SETTINGS["docker"]["container_data_target"]).resolve()
    dataraw_dir = data_dir / "raw"
    
    # Sécurité : Vérification du nom de fichier
    filename_path = Path(filename)

    if filename_path.name != filename:
        raise ValueError(
            f"Le paramètre --file doit contenir uniquement un nom de fichier : "
            f"{filename}"
        )

    # Vérification de l'extension
    filename_lower = filename.lower()

    is_processing = filename_lower.endswith(".csv.processing")
    is_csv = filename_lower.endswith(".csv")
    
    if not is_processing and not is_csv:
        raise ValueError(
            "Le fichier doit être un CSV ou un CSV en cours de traitement : "
            f"{filename}"
        )

    # 4. Chemin du fichier reçu
    source_path = (dataraw_dir / filename_path.name).resolve()

    # Sécurité contre ../
    if dataraw_dir not in source_path.parents:
        raise ValueError(
            "Chemin de fichier invalide, hors du dossier data autorisé : "
            f"{filename}"
        )
    
    # Vérifier que le fichier source existe
    if not source_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {source_path}"
        )

    if not source_path.is_file():
        raise ValueError(
            f"Le chemin indiqué n'est pas un fichier : {source_path}"
        )

    # Si .processing : renommer en .csv
    if is_processing:
        csv_filename = filename[:-len(".processing")]
        csv_path = (dataraw_dir / csv_filename).resolve()

        logger.info({
            "event": "processing_file_detected",
            "source": str(source_path),
            "target": str(csv_path),
        })

        # Éviter d'écraser un fichier CSV existant
        if csv_path.exists():
            raise FileExistsError(
                "Le fichier CSV cible existe déjà : "
                f"{csv_path}"
            )

        source_path.rename(csv_path)

        logger.info({
            "event": "processing_file_renamed",
            "source": str(source_path),
            "target": str(csv_path),
        })

    else:
        csv_path = source_path

    # Vérification finale
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier CSV introuvable après résolution : {csv_path}"
        )

    logger.info({
        "event": "csv_path_resolved",
        "data_dir": str(dataraw_dir),
        "filename": filename,
        "csv_path": str(csv_path),
    })

    return csv_path


def main():

    # -------------------------------------------------------------------------
    # 1. Récupération du fichier transmis par Airflow
    # -------------------------------------------------------------------------
    args = parse_args()

    try:
        csv_path = resolve_csv_path(args.file)
    except Exception as e:
        logger.error({
            "event": "invalid_csv_path",
            "file": args.file,
            "error": str(e),
        })

        print(json.dumps({
            "status": "failed",
            "error": str(e),
        }))

        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Vérification de l'existence du fichier
    # -------------------------------------------------------------------------
    if not csv_path.exists():
        error = (
            f"Fichier CSV introuvable dans le conteneur : "
            f"{csv_path}"
        )

        logger.error({
            "event": "csv_not_found",
            "csv_path": str(csv_path),
        })

        print(json.dumps({
            "status": "failed",
            "error": error,
        }))

        sys.exit(1)

    if not csv_path.is_file():
        error = (
            f"Le chemin indiqué n'est pas un fichier : "
            f"{csv_path}"
        )

        logger.error({
            "event": "csv_not_file",
            "csv_path": str(csv_path),
        })

        print(json.dumps({
            "status": "failed",
            "error": error,
        }))

        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Charger la configuration PostgreSQL
    # -------------------------------------------------------------------------
    try:
        cfg = load_postgres_config()
    except ConfigError as e:
        logger.error({
            "event": "config_error",
            "error": str(e),
        })

        print(json.dumps({
            "status": "failed",
            "error": str(e),
        }))

        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Lancer l'intégration
    # -------------------------------------------------------------------------
    try:
        start = time.time()

        logger.info({
            "event": "integration_start",
            "csv_path": str(csv_path),
            "host": cfg.host,
        })

        loader = WeatherDataLoader(cfg)

        result = loader.load_csv(str(csv_path))

        duration = time.time() - start

        # ---------------------------------------------------------------------
        # 5. Vérifier le résultat
        # ---------------------------------------------------------------------                                              
        if not isinstance(result, dict):
            raise Exception(
                "Résultat d'import invalide : attendu un dict"
            )
                    
        if "status" not in result:
            raise Exception(
                "Champ 'status' manquant dans le résultat d'import"
            )

        status = result["status"]

        # ---------------------------------------------------------------------
        # 6. Import échoué
        # ---------------------------------------------------------------------
        if status == "failed":
            if "error" not in result:
                raise Exception(
                    "Import échoué mais champ 'error' manquant"
                )

            raise Exception(
                f"Import FAILED : {result['error']}"
            )

        # ---------------------------------------------------------------------
        # 7. Import réussi
        # ---------------------------------------------------------------------
        if status == "success":    
            if "rows_imported" not in result:
                raise Exception(
                    "Import réussi mais champ 'rows_imported' manquant"
                )
                                 
            if "archive_path" not in result:
                raise Exception(
                    "Import réussi mais champ 'archive_path' manquant"
                )

            logger.info({
                "event": "integration_success",
                "csv_path": str(csv_path),
                "rows_imported": result["rows_imported"],
                "archive_path": str(result["archive_path"]),
                "duration_seconds": round(duration, 2),
            })

            # Important pour récupérer le résultat côté Airflow/XCom
            print(json.dumps({
                "status": "success",
                "rows_imported": result["rows_imported"],
                "archive_path": str(result["archive_path"]),
            }))

            return result

        # ---------------------------------------------------------------------
        # 8. Status inconnu
        # ---------------------------------------------------------------------

        raise Exception(
            f"Valeur de status inconnue : {status}"
        )

    except Exception as e:
        logger.error(
            {
                "event": "integration_failed",
                "csv_path": str(csv_path),
                "error": str(e),
            },
            exc_info=True,
        )

        print(json.dumps({
            "status": "failed",
            "error": str(e),
        }))

        raise


if __name__ == "__main__":
    main()
