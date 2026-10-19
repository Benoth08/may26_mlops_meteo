"""
Constantes de l'API — dérivées de settings.SETTINGS
"""
from core.settings import SETTINGS
from core.params import load_params
from core.metadata import REQUIRED_COLUMNS, ALLOWED_FEATURES, DB_COLUMNS, normalize_column_name, normalize_data # noqa: F401

from datetime import datetime
from typing import Dict


PARAMS = load_params()
SOURCE = PARAMS["dataset"]["source"]
CSV_PATH = PARAMS["csv"]["path"]
POSTGRES_TABLE = PARAMS["postgres"]["table"]
API_URL = PARAMS["api"]["url"]
    
    
API_VERSION = SETTINGS["api"]["version"]
API_NAME = SETTINGS["api"]["name"]

MODEL_PATH = SETTINGS["paths"]["models"] / SETTINGS["models"]["model"]
TABLE_RAW = SETTINGS["postgres"]["table_raw"]
IMPORT_DATE_COLUMN = SETTINGS["postgres"]["importdate_column_norm"]
RUNID_COLUMN = SETTINGS["postgres"]["importrunid_column_norm"]
SOURCE_COLUMN = SETTINGS["postgres"]["importsource_column_norm"]

# Format de l'API (YYYY-MM-DD) et format de stockage de la colonne date de TABLE_RAW (texte JJ/MM/AAAA)
API_DATE_FORMAT = "%Y-%m-%d"
RAW_DATE_FORMAT = "%d/%m/%Y"


def to_raw_date(date: str) -> str:
    """Convertit une date YYYY-MM-DD au format stocké dans TABLE_RAW (lève ValueError si invalide)."""
    return datetime.strptime(date, API_DATE_FORMAT).strftime(RAW_DATE_FORMAT)
            
            
# Paramétrage des threads numpy/sklearn
THREADS_SETTINGS = SETTINGS["threads"]

# Garde-fou anti-DoS mémoire sur /predict-batch
MAX_BATCH_ROWS = SETTINGS["api_protection"]["max_batch_rows"]
 

TARGET = SETTINGS["target"]["column_norm"]
LOCATION = SETTINGS["location"]["column_norm"]
 
HIGH_MISSING_THRESHOLD = PARAMS["preprocessing"]["missing_threshold"]
SPLIT_STRATEGY = PARAMS["split"]["strategy"]

# Seuil de décision pluie / pas pluie, identique à l'évaluation du modèle
PREDICTION_THRESHOLD = float(PARAMS["evaluation"]["threshold"])

# Degrés pour chaque direction cardinale, N = 0°, sens horaire.
COMPASS_DEGREES: Dict[str, float] = SETTINGS["compass_degrees"]
 
# Valeurs considérées comme NA (CSV + Postgres) 
POSTGRES_NA_VALUES = SETTINGS["na_values"]

REPORTS_DIR = SETTINGS["paths"]["reports"]

 
# Correspondance entre les types déclarés dans settings.py et les types
# Python utilisés par Pydantic pour construire WeatherInput dynamiquement.
TYPE_MAPPING = {
    "string": str,
    "float": float,
    "integer": int,
    "bool": bool,
    "datetime": str,
}
