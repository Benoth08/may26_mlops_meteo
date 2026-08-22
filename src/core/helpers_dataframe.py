"""
===============================================================================
    WeatherAUS — Module communs de traitement des dataframes
    ---------------------------------------------------------------------------
    Ce fichier regroupe toutes les fonctions utilitaires communes pour la manipulation des dataframes :
        - build_features
        - make_dataset
        - weather_loader
        - entraînement des modèles
        - intégration Airflow / DockerOperator

    Objectif :
        Fournir des fonctions utilitaires communes pour la manipulation des dataframes.
===============================================================================
"""

from typing import Any, Dict, List, Union

import re
import unicodedata


# ============================================================
# Normalize le nom des colonnes
# ============================================================

def normalize_column(col: str) -> str:
    """
    Normalise un nom de colonne WeatherAUS.

    Transformations :
    - suppression accents
    - séparation CamelCase
    - séparation suffixes horaires (9am, 3pm)
    - snake_case
    - suppression caractères spéciaux

    Exemples :
        WindGustSpeed -> wind_gust_speed
        Humidity9am   -> humidity_9am
        RainTomorrow  -> rain_tomorrow
    """

    if col is None:
        raise ValueError("Column name cannot be None")

    # Suppression accents
    col = unicodedata.normalize("NFKD", col)
    col = "".join(
        c for c in col if not unicodedata.combining(c)
    )

    # Séparation CamelCase
    # Exemple : WindGustSpeed -> Wind_Gust_Speed
    col = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        col
    )

    # Séparation acronymes éventuels
    col = re.sub(
        r"([A-Z]+)([A-Z][a-z])",
        r"\1_\2",
        col
    )

    # Séparation horaires
    # Exemple : Humidity9am -> Humidity_9am
    col = re.sub(
        r"(\d+)(am|pm)$",
        r"_\1\2",
        col,
        flags=re.IGNORECASE
    )

    # Minuscules
    col = col.lower()

    # Espaces et tirets
    col = re.sub(
        r"[\s\-]+",
        "_",
        col
    )

    # Caractères non autorisés
    col = re.sub(
        r"[^a-z0-9_]",
        "",
        col
    )

    # Nettoyage underscores multiples
    col = re.sub(
        r"_+",
        "_",
        col
    )

    return col.strip("_")    
    
def normalize_column_constraints(
    column_constraints: dict = None
) -> dict:
    """
    Normalise les noms de colonnes du dictionnaire de contraintes.

    Exemple :
        WindGustSpeed -> wind_gust_speed
        Humidity9am   -> humidity_9am
    """

    if column_constraints is None:
        raise ValueError("Constraints cannot be None")

    return {
        (
            constraint["norm_name"]
            if "norm_name" in constraint
            else normalize_column(column)
        ): constraint
        for column, constraint in column_constraints.items()
    }

# ============================================================
# Extraction des colonnes selon leurs métadonnées
# ============================================================
def get_columns_by_metadata(
    column_constraints: Dict[str, Dict[str, Any]],
    key: Union[str, Dict[str, Any]],
    value: Any = True,
) -> List[str]:
    """
    Retourne les noms normalisés des colonnes correspondant
    aux critères metadata.

    Compatible avec l'ancienne signature :

        get_columns_by_metadata(
            COLUMN_CONSTRAINTS,
            "feature",
            True
        )

    et la nouvelle syntaxe :

        get_columns_by_metadata(
            COLUMN_CONSTRAINTS,
            {
                "feature": True,
                "type": "numeric"
            }
        )
    """

    columns = []

    # Compatibilité ancien format :
    # key="feature", value=True
    if isinstance(key, str):
        criteria = {
            key: value
        }

    # Nouveau format :
    # key={"feature": True, "type": "numeric"}
    elif isinstance(key, dict):
        criteria = key

    else:
        raise TypeError(
            "key doit être une chaîne ou un dictionnaire de critères."
        )

    for column, metadata in column_constraints.items():

        match = True

        for criterion, expected in criteria.items():

            current_value = metadata.get(criterion)

            # Gestion des valeurs multiples
            if isinstance(expected, list):
                if current_value not in expected:
                    match = False
                    break

            elif current_value != expected:
                match = False
                break

        if match:
            columns.append(
                metadata.get(
                    "norm_name",
                    normalize_column(column)
                )
            )

    return columns
    

def get_all_columns(
    column_constraints: dict
) -> List[str]:
    """
    Retourne un dictionnaire où les clés sont les noms normalisés des colonnes
    (norm_name si présent, sinon normalize_column()).

    """

    return [
        normalize_column(column)
        for column, metadata in column_constraints.items()
    ]

    
def get_required_columns(
    column_constraints: dict
) -> List[str]:
    """
     Retourne les colonnes obligatoires du schéma.

    Une colonne est obligatoire si :
        "required": True
    """

    return get_columns_by_metadata(column_constraints, "required")


def get_feature_columns(
    column_constraints: dict
) -> List[str]:
    """
    Retourne les colonnes utilisées comme features ML.

    Une colonne est considérée comme une feature si :
        "feature": True
    """

    return get_columns_by_metadata(column_constraints, "feature")

def get_numeric_columns(
    column_constraints: dict
) -> List[str]:
    """
    Retourne les colonnes numériques du schéma.

    Une colonne est considérée comme numérique si :
        "type": Float ou Int
    """

    return get_columns_by_metadata(column_constraints, "type", value=["float", "int"])

    
def get_technical_columns(
    column_constraints: dict
) -> List[str]:
    """
    Retourne les colonnes techniques du schéma.

    Une colonne est considérée comme technique si :
        "technical": True
    """

    return get_columns_by_metadata(column_constraints, "technical")
    
def get_wind_direction_columns(
    column_constraints: dict
) -> List[str]:
    """
    Retourne les colonnes de direction de vent du schéma.

    Une colonne est considérée comme une direction de vent si :
        "wind_direction": True
    """

    return get_columns_by_metadata(column_constraints, "wind_direction")   
