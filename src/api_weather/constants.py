"""
Constantes de l'API — dérivées de settings.SETTINGS
"""
from core.settings import SETTINGS
from core.metadata import COLUMN_CONSTRAINTS, REQUIRED_COLUMNS, TECHNICAL_COLUMNS, WIND_DIRECTION_COLUMNS, CATEGORICAL_COLUMNS, ALLOWED_FEATURES, FEATURE_COLUMNS, NUMERIC_COLUMNS, DB_COLUMNS, normalize_column_name, normalize_data

from typing import Dict


API_VERSION = SETTINGS["api"]["version"]
API_NAME = SETTINGS["api"]["name"]

MODEL_PATH = SETTINGS["paths"]["models"] / SETTINGS["models"]["model"]
TABLE_RAW = SETTINGS["postgres"]["table_raw"]
IMPORT_DATE_COLUMN = SETTINGS["postgres"]["importdate_column_norm"]
RUNID_COLUMN = SETTINGS["postgres"]["importrunid_column_norm"]
SOURCE_COLUMN = SETTINGS["postgres"]["importsource_column_norm"]
            
            
# Paramétrage des threads numpy/sklearn
THREADS_SETTINGS = SETTINGS["threads"]

# Garde-fou anti-DoS mémoire sur /predict-batch
MAX_BATCH_ROWS = SETTINGS["api_protection"]["max_batch_rows"]
 

TARGET = SETTINGS["target"]["column_norm"]
LOCATION = SETTINGS["location"]["column_norm"]
 
HIGH_MISSING_THRESHOLD = SETTINGS["missing_threshold"]
SPLIT_STRATEGY = SETTINGS["split_strategy"]

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
