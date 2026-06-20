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

from scripts.weather_loader import WeatherDataLoader

import os
import sys
import json
import time


def main():

    # -----------------------------
    # 1. Lire les variables d'env
    # ----------------------------- 
    db = os.getenv("POSTGRES_WTH_DB")
    user = os.getenv("POSTGRES_WTH_USER")
    pwd = os.getenv("POSTGRES_WTH_PASSWORD")

    # Vérification explicite
    if not all([db, user, pwd]):
        print(json.dumps({
            "status": "failed",
            "error": "Missing environment variables",
            "POSTGRES_WTH_DB": db,
            "POSTGRES_WTH_USER": user,
            "POSTGRES_WTH_PASSWORD": pwd
        }))
        sys.exit(1)

    # -----------------------------
    # 2. Lancer l’integration
    # -----------------------------
    try:
        
        start = time.time()

        loader = WeatherDataLoader(
            host="weather-postgres",
            port=5432,
            database=db,
            user=user,
            password=pwd
        )
        
        result = loader.load_csv("/data/weatherAUS.csv")
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

            # Log succès
            print(json.dumps({
                "status": "success",
                "rows_imported": result["rows_imported"],
                "archive_path": result["archive_path"]
            }))

            return result

        # -----------------------------
        # 5. Cas : status inconnu
        # -----------------------------
        raise Exception(f"Valeur de status inconnue : {status}")

    except Exception as e:
        print(json.dumps({"status": "failed", "error": str(e)}))
        raise


if __name__ == "__main__":
    main()
    
