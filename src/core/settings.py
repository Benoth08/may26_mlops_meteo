"""
===============================================================================
    WeatherAUS — Module central de configuration
    ---------------------------------------------------------------------------
    Ce fichier regroupe toutes les constantes globales utilisées dans :
        - build_features
        - make_dataset
        - weather_loader
        - entraînement des modèles
        - intégration Airflow / DockerOperator

    Objectif :
        Fournir un point de vérité unique, lisible et maintenable.
===============================================================================
"""

from pathlib import Path


# ============================================================================
# Paramètres généraux du projet
# ============================================================================
PROJECT_NAME = "WeatherAUS"
VERSION = "1.0.0"
RANDOM_SEED = 42

API_VERSION = "1.0.0"
API_NAME = "weather-api"

# ============================================================================
# Logging, Debud...
# ============================================================================
LOGGING_LEVEL = "INFO"


# ============================================================================
# Valeurs NA reconnues (CSV + Postgres)
# ============================================================================
NA_VALUES = [
    "", " ", "NA", "N/A", "na", "n/a",
    "NULL", "null", "None", "none",
    "NaN", "nan", "?", "--"
]


# ============================================================================
# Contraintes de schéma (types, ranges, valeurs autorisées)
# ============================================================================
COLUMN_CONSTRAINTS = {
    "Date": {"norm_name": "date", "type": "datetime", "nullable": False, "required": True, "technical":False, "feature": True, "wind_direction": False, "description": "Date de la mesure météo"},
    "Location": {"norm_name": "location","type": "string", "nullable": False, "required": True, "technical":False, "feature": True, "wind_direction": False, "allowed_values": None, "description": "Ville ou station météo"},

    "MinTemp": {"norm_name": "min_temp", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (-50, 60), "description": "Température minimale du jour"},
    "MaxTemp": {"norm_name": "max_temp", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (-50, 60), "description": "Température maximale du jour"},
    "Temp9am": {"norm_name": "temp_9am", "type": "float","nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (-50, 60), "description": "Température à 9h"},
    "Temp3pm": {"norm_name": "temp_3pm", "type": "float", "nullable": True,"required": True, "technical":False, "feature": True, "wind_direction": False, "range": (-50, 60), "description": "Température à 15h"},

    "Rainfall": {"norm_name": "rainfall", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 500), "description": "Pluviométrie en mm"},
    "Evaporation": {"norm_name": "evaporation", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 50), "description": "Évaporation"},
    "Sunshine": {"norm_name": "sunshine", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 15), "description": "Durée d'ensoleillement"},

    "Humidity9am": {"norm_name": "humidity_9am", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 100), "description": "Humidité à 9h"},
    "Humidity3pm": {"norm_name": "humidity_3pm", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 100), "description": "Humidité à 15h"},

    "Pressure9am": {"norm_name": "pressure_9am", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (900, 1100), "description": "Pression à 9h"},
    "Pressure3pm": {"norm_name": "pressure_3pm", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (900, 1100), "description": "Pression à 15h"},

    "WindGustDir": {"norm_name": "wind_gust_dir", "type": "string", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": True, "allowed_values": ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"], "description": "Direction rafale"},
    "WindGustSpeed": {"norm_name": "wind_gust_speed", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 200), "description": "Vitesse rafale"},
    "WindDir9am": {"norm_name": "wind_dir_9am", "type": "string", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": True, "allowed_values": ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"], "description": "Direction vent à 9h"},
    "WindDir3pm": {"norm_name": "wind_dir_3pm", "type": "string", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": True, "allowed_values": ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"], "description": "Direction vent à 15h"},
    "WindSpeed9am": {"norm_name": "wind_speed_9am", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 150), "description": "Vitesse vent à 9h"},
    "WindSpeed3pm": {"norm_name": "wind_speed_3pm", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 150), "description": "Vitesse vent à 15h"},

    "Cloud9am": {"norm_name": "cloud_9am", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 9), "description": "Couverture nuageuse à 9h"},
    "Cloud3pm": {"norm_name": "cloud_3pm", "type": "float", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "range": (0, 9), "description": "Couverture nuageuse à 15h"},

    "RainToday": {"norm_name": "rain_today", "type": "string", "nullable": True, "required": True, "technical":False, "feature": True, "wind_direction": False, "allowed_values": ["Yes", "No"], "description": "Pluie observée aujourd'hui"},
    "RainTomorrow": {"norm_name": "rain_tomorrow", "type": "string", "nullable": True, "required": True, "technical":False, "feature": False, "wind_direction": False, "allowed_values": ["Yes", "No"], "description": "Pluie prévue demain (cible)"},
    
    "date_import": {"norm_name": "date_import", "type": "datetime", "nullable": False, "required": False, "technical":True, "feature": False, "wind_direction": False, "description": "Date de l'import"},
    "data_source": {"norm_name": "data_source", "type": "string", "nullable": False, "required": False, "technical":True, "feature": False, "wind_direction": False, "description": "Source de l'import"},
    "run_id": {"norm_name": "run_id", "type": "string", "nullable": False, "required": False, "technical":True, "feature": False, "wind_direction": False, "description": "Identifiant du run d'import"}
}


# ============================================================================
# Colonne métier cible de prévision
# ============================================================================
TARGET_COLUMN = "RainTomorrow"
TARGET_COLUMN_NORM = "rain_tomorrow"

# ============================================================================
# Colonnes Location
# ============================================================================

LOCATION_COLUMN = "Location"
LOCATION_COLUMN_NORM = "location"

# ============================================================================
# Import & Clean des données
# ============================================================================

POSTGRES_TABLE_RAW = "weather_data_raw"
IMPORT_DATE_COLUMN_NORM = "date_import"
IMPORT_RUNID_COLUMN_NORM = "run_id"
IMPORT_SOURCE_COLUMN_NORM = "data_source"

POSTGRES_TABLE_CLEAN = "weather_data_clean"
CLEAN_DATE_COLUMN_NORM = "date_clean"
CLEAN_RUNID_COLUMN_NORM = "run_id"
CLEAN_SOURCE_COLUMN_NORM = "data_source"

POSTGRES_TABLE_FEATURES_ML = "weather_features_ml"

# ============================================================================
# Directions de vent + encodage cyclique
# ============================================================================
WIND_DIRECTION_COLUMNS = ["WindGustDir", "WindDir9am", "WindDir3pm"]

COMPASS_DEGREES = {
    direction: i * (360.0 / 16)
    for i, direction in enumerate([
        "N","NNE","NE","ENE","E","ESE","SE","SSE",
        "S","SSW","SW","WSW","W","WNW","NW","NNW"
    ])
}


# ============================================================================
# Paramètres de preprocessing
# ============================================================================
HIGH_MISSING_THRESHOLD = 0.30

IMPUTATION_STRATEGY = {
    "numeric": "iterative",
    "categorical": "mode",
    "location": "target_encoder",
}

SCALER = "robust"  # robust / standard / minmax


# ============================================================================
# Paramètres de split
# ============================================================================
SPLIT_STRATEGY = "temporal"
TEST_SIZE = 0.20


# ============================================================================
# Paramètres d’exécution ML (threads)
# ============================================================================
OMP_NUM_THREADS = 1
MKL_NUM_THREADS = 1
NUMEXPR_NUM_THREADS = 1

# ============================================================================
# Paramètres de protection des requêtes vers API
# ============================================================================
MAX_BATCH_ROWS = 50_000

# ============================================================================
# Paramètres Postgres
# !!! user/password ne doivent jamais avoir de valeurs renseignées en dur !!!
# ============================================================================
POSTGRES_DEFAULT_HOST = "localhost"
POSTGRES_DEFAULT_PORT = 5432
POSTGRES_DEFAULT_DB = "weather"
POSTGRES_DEFAULT_USR = None
POSTGRES_DEFAULT_PWD = None

CSV_CHUNK_SIZE = 5000
SQL_BATCH_SIZE = 500

# ============================================================================
# Dossiers du projet
# ============================================================================
DATA_DIR = Path("/data")
PROCESSED_DIR = DATA_DIR / "processed"
ARCHIVE_DIR = DATA_DIR / "archive"
MODELS_DIR = Path("/models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = Path("/metrics")
LOGS_DIR = Path("logs")


# ============================================================================
# Modele
# ============================================================================
MODEL_NAME = "model.joblib"
MODEL_PKL_NAME = "model.pkl"
REGISTERED_MODEL_NAME = "weather-rain-model"
RAW_DATA = "weatherAUS.csv"
PREPROCESSED_DATA = "weather_data_clean.csv"
DATASET_NAME = "dataset.joblib"
METRICS_NAME = "scores.json"
PREDICTIONS_NAME = "predictions.csv"
PREPROCESSOR_NAME = "preprocessor.joblib"
BEST_PARAMS_NAME = "best_params.joblib"
REPORTS_NAME = "missing_values_report.csv" 


# ============================================================================
# Objet global SETTINGS
# ============================================================================
SETTINGS = {
    "project": PROJECT_NAME,
    "version": VERSION,
    "seed": RANDOM_SEED,
    
    "api": {
        "version": API_VERSION,
        "name": API_NAME
    },
    
    "logging": {
        "level": LOGGING_LEVEL
    },

    "target": {
        "column": TARGET_COLUMN,
        "column_norm": TARGET_COLUMN_NORM 
    },
    
    "location": {
        "column": LOCATION_COLUMN,
        "column_norm": LOCATION_COLUMN_NORM 
    },
    
    "na_values": NA_VALUES,
    "constraints": COLUMN_CONSTRAINTS,

    "wind_columns": WIND_DIRECTION_COLUMNS,
    "compass_degrees": COMPASS_DEGREES,

    "missing_threshold": HIGH_MISSING_THRESHOLD,
    "imputation": IMPUTATION_STRATEGY,
    "scaler": SCALER,

    "split_strategy": SPLIT_STRATEGY,
    "test_size": TEST_SIZE,

    "threads": {
        "OMP_NUM_THREADS": OMP_NUM_THREADS,
        "MKL_NUM_THREADS": MKL_NUM_THREADS,
        "NUMEXPR_NUM_THREADS": NUMEXPR_NUM_THREADS,
    },
    
    "api_protection": {
        "max_batch_rows" : MAX_BATCH_ROWS,
    },

    "postgres": {
        "default_host": POSTGRES_DEFAULT_HOST,
        "default_port": POSTGRES_DEFAULT_PORT,
        "default_db": POSTGRES_DEFAULT_DB,
        "default_user": POSTGRES_DEFAULT_USR,
        "default_password": POSTGRES_DEFAULT_PWD,
        "table_raw": POSTGRES_TABLE_RAW,
        "importdate_column_norm": IMPORT_DATE_COLUMN_NORM,
        "importrunid_column_norm": IMPORT_RUNID_COLUMN_NORM,
        "importsource_column_norm": IMPORT_SOURCE_COLUMN_NORM,
        "table_clean": POSTGRES_TABLE_CLEAN,
        "cleandate_column_norm": CLEAN_DATE_COLUMN_NORM,
        "cleanrunid_column_norm": CLEAN_RUNID_COLUMN_NORM,
        "cleansource_column_norm": CLEAN_SOURCE_COLUMN_NORM,
        "table_features": POSTGRES_TABLE_FEATURES_ML,
        "csv_chunk_size": CSV_CHUNK_SIZE,
        "sql_batch_size": SQL_BATCH_SIZE
    },

    "paths": {
        "data": DATA_DIR,
        "processed": PROCESSED_DIR,
        "archive": ARCHIVE_DIR,
        "models": MODELS_DIR,
        "reports": REPORTS_DIR,
        "figures": FIGURES_DIR,
        "metrics": METRICS_DIR,
        "logs": LOGS_DIR,
    },
    
    "models": {
        "model" :MODEL_NAME,
        "model_pkl": MODEL_PKL_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "rawdata": RAW_DATA,
        "preprocessed_data": PREPROCESSED_DATA,
        "dataset": DATASET_NAME,
        "preprocessor": PREPROCESSOR_NAME,
        "best_params": BEST_PARAMS_NAME,
        "reports": REPORTS_NAME,
        "metrics": METRICS_NAME,
        "predictions": PREDICTIONS_NAME,
    }
    
}
