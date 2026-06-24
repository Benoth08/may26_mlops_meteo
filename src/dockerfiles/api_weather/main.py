#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Weather API / Predict next-day rain in Australia
    
    Description :
        API REST proposant 1 endpoint :
        - Vérifier si l'API est opérationnelle

    Version :
        1.0.0

    Historique :
        2026-06-11  -  Création du module
===============================================================================
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from datetime import datetime

import psycopg2
import os


"""
===============================================================================
Init FastAPI
===============================================================================
"""

app = FastAPI(
    title="Projet Weather API",
    description="API permettant de prédire si la journée du lendemain sera pluvieuse",
    version="1.0.0"
)


"""
===============================================================================
Endpoint 1 : /health - Vérifie que l'API est fonctionnelle (méthode GET)
===============================================================================
"""

responsesEP1 = {
    200: {"description": "OK"}
}
@app.get('/health', responses=responsesEP1)
def get_():
    """
    Vérifie que l'API est fonctionnelle
    """
    return {
        "status": "ok",
        "service": "weather-api",
        "version": "1.0.0"
    }
    

"""
===============================================================================
Endpoint 2 : /last-import - Fournit les informations concernant le dernier imporrt de données (méthode GET)
===============================================================================
"""

responsesEP_LI = {
    200: {"description": "OK"},
    404: {"description": "Aucun import disponible"},
    500 : {"description": "Erreur serveur"}
}
@app.get("/last-import", responses=responsesEP_LI)
def get_last_import():
    """
    Retourne les informations du dernier import :
    - date d'import
    - fichier source
    - nombre de lignes importées
    """

    # -----------------------------
    # 1. Lire les variables d'env
    # ----------------------------- 
    db = os.getenv("POSTGRES_WTH_DB")
    user = os.getenv("POSTGRES_WTH_USER")
    pwd = os.getenv("POSTGRES_WTH_PASSWORD")

    # Vérification explicite
    if not all([db, user, pwd]):
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "error": "Missing environment variables",
                "POSTGRES_WTH_DB": db,
                "POSTGRES_WTH_USER": user,
                "POSTGRES_WTH_PASSWORD": pwd
            }
        )

    # -----------------------------
    # 2. Lancer l’integration
    # -----------------------------

    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_WTH_HOST", "weather-postgres"),
            port=os.getenv("POSTGRES_WTH_PORT", 5432),
            database=db,
            user=user,
            password=pwd
        )
        cur = conn.cursor()

        # Récupérer le dernier import
        cur.execute("""
            SELECT 
                import_date,
                source_file,
                COUNT(*) AS rows_imported
            FROM weather_data_raw
            WHERE import_date = (
                SELECT MAX(import_date) FROM weather_data_raw
            )
            GROUP BY import_date, source_file
            ORDER BY import_date DESC
            LIMIT 1;
        """)

        row = cur.fetchone()
        cur.close()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------
    # 4. Aucun résultat
    # -----------------------------      
    if not row:
        raise HTTPException(
            status_code=404,
            detail={
                "status": "failed",
                "error": "Aucun import trouvé"
            }
        )

    import_date, source_file, rows_imported = row

    return {
        "status": "success",
        "import_date": import_date.isoformat(),
        "source_file": source_file,
        "rows_imported": rows_imported
    }




"""
===============================================================================
Endpoint 3 : /weather - fournit une donnée météo pour une date donnée (méthode GET)
===============================================================================
"""

ALLOWED_FEATURES = {
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustSpeed",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "RainTomorrow"
}

responsesEP_WTH_FT = {
    200: {"description": "OK"}
}
@app.get("/weather/features", responses=responsesEP_WTH_FT)
def get_weather_features():
    """
    Retourne la liste des paramètres météo autorisés.
    """
    return {
        "status": "success",
        "count": len(ALLOWED_FEATURES),
        "features": sorted(list(ALLOWED_FEATURES))
    }




responsesEP_WTH = {
    200: {"description": "OK"},
    400: {"description": "Paramètre incorrect (date ou champ météo)"},
    404: {"description": "Aucun import disponible"},
    500 : {"description": "Erreur serveur"}
}
@app.get("/weather", responses=responsesEP_WTH)
def get_weather(
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
    location: str = Query(..., description="Nom de la station météo"),
    feature: str = Query("Rainfall", description="Nom de la colonne météo à consulter")
):
    """
    Retourne une donnée météo pour une date, une localisation et un paramétre donnés.
    """

    # -----------------------------
    # 1. Vérification des paramètres (whitelist, format date)
    # -----------------------------
    if feature not in ALLOWED_FEATURES:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "error": f"Paramètre '{feature}' non disponible",
                "allowed_features": list(ALLOWED_FEATURES)
            }
        )

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "error": f"Format de date invalide : '{date}'. Format attendu : YYYY-MM-DD"
            }
        )

    # -----------------------------
    # 2. Vérification variables d'env
    # -----------------------------
    db = os.getenv("POSTGRES_WTH_DB")
    user = os.getenv("POSTGRES_WTH_USER")
    pwd = os.getenv("POSTGRES_WTH_PASSWORD")

    if not all([db, user, pwd]):
        raise HTTPException(
            status_code=500,
            detail="Missing environment variables"
        )

    # -----------------------------
    # 3. Connexion DB
    # -----------------------------
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_WTH_HOST", "weather-postgres"),
            port=os.getenv("POSTGRES_WTH_PORT", 5432),
            database=db,
            user=user,
            password=pwd
        )
        cur = conn.cursor()

        query = f"""
            SELECT {feature}
            FROM weather_data_raw
            WHERE date = %s
              AND location = %s
            ORDER BY import_date DESC
            LIMIT 1;
        """
        
        cur.execute(query, [date, location])
        row = cur.fetchone()

        cur.close()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------
    # 4. Aucun résultat
    # -----------------------------
    if not row:
        raise HTTPException(
            status_code=404,
            detail="Aucune donnée trouvée pour cette date / station"
        )

    value = row[0]

    return {
        "status": "success",
        "date": date,
        "location": location,
        "feature": feature,
        "value": value
    }


responsesEP_WTH_ALL = {
    200: {"description": "OK"},
    400: {"description": "Paramètre météo indisponible"},
    404: {"description": "Aucun import disponible"},
    500 : {"description": "Erreur serveur"}
}
@app.get("/weather/all-features", responses=responsesEP_WTH_ALL)
def get_weather_all_features(
    date: str = Query(..., description="Date au format YYYY-MM-DD"),
    location: str = Query(..., description="Nom de la station météo (obligatoire)")
):
    """
    Retourne tous les paramètres météo pour une date et une station données.
    """

    # -----------------------------
    # 1. Vérification des paramètres (format date)
    # ----------------------------
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "failed",
                "error": f"Format de date invalide : '{date}'. Format attendu : YYYY-MM-DD"
            }
        )


    # -----------------------------
    # 1. Vérification variables d'env
    # -----------------------------
    db = os.getenv("POSTGRES_WTH_DB")
    user = os.getenv("POSTGRES_WTH_USER")
    pwd = os.getenv("POSTGRES_WTH_PASSWORD")

    if not all([db, user, pwd]):
        raise HTTPException(
            status_code=500,
            detail="Missing environment variables"
        )

    # -----------------------------
    # 2. Construire la liste SQL des colonnes autorisées
    # -----------------------------
    allowed_cols_sql = ", ".join(ALLOWED_FEATURES)
    
    
    # -----------------------------
    # 3. Connexion DB
    # -----------------------------
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_WTH_HOST", "weather-postgres"),
            port=os.getenv("POSTGRES_WTH_PORT", 5432),
            database=db,
            user=user,
            password=pwd
        )
        cur = conn.cursor()

        # -----------------------------
        # 3. Requête : toutes les colonnes + import_date
        # -----------------------------
        query = f"""
            SELECT {allowed_cols_sql}
            FROM weather_data_raw
            WHERE date = %s
              AND location = %s
            ORDER BY import_date DESC
            LIMIT 1;
        """

        cur.execute(query, [date, location])
        row = cur.fetchone()

        cur.close()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # -----------------------------
    # 4. Aucun résultat
    # -----------------------------
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnée trouvée pour {location} à la date {date}"
        )

    # -----------------------------
    # 5. Transformer en dict propre
    # -----------------------------
    data = dict(zip(ALLOWED_FEATURES, row))

    return {
        "status": "success",
        "location": location,
        "date": date,
        "data": data
    }

