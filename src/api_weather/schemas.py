"""
Schémas Pydantic de l'API. Isolés du reste : ce module ne connaît ni la DB
ni le modèle ML, uniquement la forme des données échangées avec le client.
"""
from typing import Optional

from pydantic import BaseModel, create_model

from core.metadata import FEATURE_CONSTRAINTS 
from constants import TARGET, TYPE_MAPPING, normalize_column_name


def build_weather_model(schema: dict):
    """
    Construit dynamiquement le modèle Pydantic utilisé par l'API FastAPI.

    Seules les features du modèle sont exposées en entrée API.
    La target (rain_tomorrow) est volontairement exclue.
    """

    fields = {}

    for column, metadata in schema.items():        
        # Nom normalisé
        norm_name = metadata.get(
            "norm_name",
            normalize_column_name(column)
        )
        
        # Exclusion de la cible
        if norm_name == TARGET:
            continue

        # Type de champ
        py_type = TYPE_MAPPING[metadata["type"]]

        # Non nullable (date, location) : obligatoire et typé strictement.
        # Nullable : la clé peut être absente ou null, l'imputer du pipeline
        # gère la valeur manquante (ex : sunshine souvent non mesuré).
        if metadata.get("nullable", True):
            fields[norm_name] = (Optional[py_type], None)
        else:
            fields[norm_name] = (py_type, ...)

    return create_model("WeatherInput", __base__=BaseModel, **fields)


# Schéma d'entrée pour /predict
WeatherInput = build_weather_model(FEATURE_CONSTRAINTS)
