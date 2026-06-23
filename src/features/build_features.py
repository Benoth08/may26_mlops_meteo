"""
Module de prétraitement des données WeatherAUS.

Objectif :
Préparer les données pour prédire la variable cible RainTomorrow.

Ce module couvre :
1.  Chargement des données (CSV, PostgreSQL, API REST)
2.  Validation du schéma
3.  Rapport des valeurs manquantes
4.  Identification des colonnes avec plus de 30 % de valeurs manquantes
5.  Nettoyage et encodage de la cible
6.  Encodage de RainToday en binaire (0/1)
7.  Parsing de la date
8.  Feature engineering temporel
9.  Encodage cyclique (mois, jour de l'année, directions de vent)
10. Feature engineering météo
11. Séparation X / y (colonnes techniques exclues des features)
12. Split train/test
13. Construction du préprocesseur scikit-learn
14. Prévention de la fuite de données

Sources de données supportées :
- CSV local (défaut, rétrocompatible)
- PostgreSQL via SQLAlchemy (nécessite sqlalchemy, psycopg2-binary)
- API REST JSON (nécessite requests)

Stratégie de traitement des valeurs manquantes :
- Numériques  : IterativeImputer (modélise chaque variable à partir des autres)
                + indicateur de missingness + RobustScaler (robuste aux outliers)
- Location    : TargetEncoder (probabilité de pluie par ville, 1 colonne vs 49)
                Fallback vers OneHotEncoder si TargetEncoder indisponible (sklearn < 1.3).
- Catégorielles restantes : imputation mode + OneHotEncoder

Le préprocesseur est retourné non entraîné.
Il doit être intégré dans un Pipeline scikit-learn avec le modèle final.

Colonnes techniques (date_import, data_source, run_id) :
- Conservées dans les données brutes pour le monitoring Airflow.
- Exclues de X (features du modèle) par drop_unused_columns et split_features_target.
"""

import argparse
import logging

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# Constantes
# ============================================================

TARGET = "RainTomorrow"

DEFAULT_DATA_PATH = r"C:\Users\ander\OneDrive\Documents\Projet_MLOPs\weatherAUS.csv"

HIGH_MISSING_THRESHOLD = 0.30

# Colonnes de traçabilité Airflow/PostgreSQL : conservées pour le monitoring
# mais systématiquement exclues des features du modèle.
TECHNICAL_COLUMNS = ["date_import", "data_source", "run_id"]

REQUIRED_COLUMNS = [
    "Date",
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
    "RainTomorrow",
]

WIND_DIRECTION_COLUMNS = ["WindGustDir", "WindDir9am", "WindDir3pm"]

# Degrés pour chaque direction cardinale, N = 0°, sens horaire.
COMPASS_DEGREES: Dict[str, float] = {
    direction: i * (360.0 / 16)
    for i, direction in enumerate([
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ])
}


# ============================================================
# Chargement des données
# ============================================================

def load_data_from_csv(data_path: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un fichier CSV.

    Parameters
    ----------
    data_path : str
        Chemin vers le fichier CSV.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {data_path}")

    logger.info("Chargement CSV depuis %s", data_path)
    return pd.read_csv(path)


def load_data_from_postgres(connection_uri: str, table_name: str) -> pd.DataFrame:
    """
    Charge le dataset depuis une table PostgreSQL via SQLAlchemy.

    Parameters
    ----------
    connection_uri : str
        URI de connexion SQLAlchemy.
        Exemple : postgresql://user:pwd@localhost:5432/db_name
    table_name : str
        Nom de la table à lire.

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    Dépendances requises : sqlalchemy, psycopg2-binary.
    Les colonnes techniques date_import, data_source, run_id seront présentes
    si elles existent en base ; elles seront exclues des features du modèle.
    """
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy est requis pour charger depuis PostgreSQL. "
            "Installez-le avec : pip install sqlalchemy psycopg2-binary"
        ) from exc

    logger.info("Connexion PostgreSQL, lecture de la table '%s'", table_name)
    engine = create_engine(connection_uri)
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)  # noqa: S608


def load_data_from_api(api_url: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un endpoint REST retournant du JSON.

    Deux formats de réponse supportés :
    - Liste JSON directe  : [{...}, {...}]
    - Objet avec clé data : {"data": [{...}, {...}], ...}

    Parameters
    ----------
    api_url : str
        URL complète de l'endpoint (ex. http://localhost:8057/meteo).

    Returns
    -------
    pd.DataFrame

    Notes
    -----
    Dépendance requise : requests.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "requests est requis pour charger depuis l'API. "
            "Installez-le avec : pip install requests"
        ) from exc

    logger.info("Requête API : %s", api_url)
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload["data"])
    return pd.DataFrame(payload)


def load_dataset(
    source: str = "csv",
    data_path: Optional[str] = None,
    connection_uri: Optional[str] = None,
    table_name: Optional[str] = None,
    api_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Charge le dataset depuis la source spécifiée.

    Par défaut, utilise le CSV local pour préserver la compatibilité existante.

    Parameters
    ----------
    source : {"csv", "postgres", "api"}
        Source de données. Par défaut : "csv".
    data_path : str, optional
        Chemin CSV. Requis si source="csv". Utilise DEFAULT_DATA_PATH si absent.
    connection_uri : str, optional
        URI SQLAlchemy. Requis si source="postgres".
    table_name : str, optional
        Nom de la table PostgreSQL. Requis si source="postgres".
    api_url : str, optional
        URL de l'endpoint REST. Requis si source="api".

    Returns
    -------
    pd.DataFrame
    """
    if source == "csv":
        return load_data_from_csv(data_path or DEFAULT_DATA_PATH)

    if source == "postgres":
        if not connection_uri or not table_name:
            raise ValueError(
                "source='postgres' requiert connection_uri et table_name."
            )
        return load_data_from_postgres(connection_uri, table_name)

    if source == "api":
        if not api_url:
            raise ValueError("source='api' requiert api_url.")
        return load_data_from_api(api_url)

    raise ValueError(
        f"Source inconnue : '{source}'. Valeurs acceptées : csv, postgres, api."
    )


def load_data(data_path: str) -> pd.DataFrame:
    """Alias rétrocompatible de load_data_from_csv."""
    return load_data_from_csv(data_path)


# ============================================================
# Validation du schéma
# ============================================================

def validate_schema(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> None:
    """
    Vérifie que le dataset contient toutes les colonnes attendues.

    Raises
    ------
    ValueError
        Si une ou plusieurs colonnes obligatoires sont absentes.
    """
    if required_columns is None:
        required_columns = REQUIRED_COLUMNS

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Le dataset ne respecte pas le schéma attendu. "
            f"Colonnes manquantes : {missing_columns}"
        )


# ============================================================
# Rapport des valeurs manquantes
# ============================================================

def build_missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un rapport détaillé sur les valeurs manquantes.

    Returns
    -------
    pd.DataFrame
        Rapport contenant le taux, le nombre de valeurs manquantes,
        le type de variable et le nombre de valeurs distinctes.
    """
    report = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
            "missing_rate_pct": (df.isna().mean().values * 100).round(2),
            "unique_values": df.nunique(dropna=True).values,
        }
    )

    report = report.sort_values("missing_rate", ascending=False)
    report = report.reset_index(drop=True)

    return report


def save_missing_values_report(
    report: pd.DataFrame,
    output_path: str = "reports/missing_values_report.csv",
) -> None:
    """Sauvegarde le rapport des valeurs manquantes."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(output_file, index=False)


def identify_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD,
    exclude_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Identifie les colonnes dont le taux de valeurs manquantes dépasse le seuil.

    Par défaut, le seuil est 30 %.

    Important :
    Ces colonnes ne sont pas supprimées automatiquement.
    La stratégie principale consiste à les conserver avec imputation et
    indicateur de valeur manquante.
    """
    if exclude_columns is None:
        exclude_columns = []

    missing_rate = df.isna().mean()

    high_missing_columns = (
        missing_rate[missing_rate > threshold]
        .index
        .difference(exclude_columns)
        .tolist()
    )

    return high_missing_columns


# ============================================================
# Nettoyage de la cible
# ============================================================

def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et encode la variable cible RainTomorrow.

    Traitements :
    - suppression des lignes où RainTomorrow est manquante ;
    - encodage binaire : No = 0, Yes = 1.
    """
    df = df.copy()

    df = df.dropna(subset=[TARGET])

    target_mapping = {"No": 0, "Yes": 1}
    df[TARGET] = df[TARGET].map(target_mapping)

    if df[TARGET].isna().any():
        raise ValueError(
            "La variable cible contient des modalités inattendues. "
            "Modalités attendues : 'Yes' et 'No'."
        )

    df[TARGET] = df[TARGET].astype(int)

    return df


def encode_rain_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode RainToday en binaire (0/1) pour cohérence avec RainTomorrow.

    Les valeurs manquantes restent NaN et seront traitées par l'imputer.
    Evite les 2 colonnes OHE et traite RainToday comme une variable numérique.
    """
    df = df.copy()
    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})
    return df


# ============================================================
# Traitement de la date
# ============================================================

def parse_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit la colonne Date en datetime.

    Les dates invalides sont transformées en NaT puis supprimées.
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des variables temporelles à partir de Date."""
    df = df.copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute un encodage cyclique pour Month et DayOfYear.

    Cet encodage permet de mieux représenter la saisonnalité.
    Par exemple, décembre et janvier sont proches dans le temps.
    """
    df = df.copy()

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["DayOfYear_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["DayOfYear_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)

    return df


def encode_wind_directions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes de direction de vent en encodage cyclique sin/cos.

    Les 16 directions cardinales (N, NNE, …, NNW) sont mappées en degrés
    puis projetées sur sin/cos pour préserver la continuité circulaire :
    N (0°) et NNW (337.5°) sont voisins, ce qu'OHE ne peut pas capturer.
    Les colonnes string originales sont supprimées après transformation.
    """
    df = df.copy()

    for col in WIND_DIRECTION_COLUMNS:
        if col not in df.columns:
            continue

        degrees = df[col].map(COMPASS_DEGREES)
        df[f"{col}_sin"] = np.sin(np.radians(degrees))
        df[f"{col}_cos"] = np.cos(np.radians(degrees))
        df = df.drop(columns=[col])

    return df


# ============================================================
# Feature engineering météo
# ============================================================

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des variables météo supplémentaires.

    Variables créées :
    - TempRange
    - TempChange
    - HumidityDrop
    - PressureDrop
    - WindSpeedChange
    - RainfallLog1p
    - HasRainfall
    - HighHumidity3pm
    - StrongWindGust
    """
    df = df.copy()

    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["TempChange"] = df["Temp3pm"] - df["Temp9am"]

    df["HumidityDrop"] = df["Humidity9am"] - df["Humidity3pm"]
    df["PressureDrop"] = df["Pressure9am"] - df["Pressure3pm"]
    df["WindSpeedChange"] = df["WindSpeed3pm"] - df["WindSpeed9am"]

    df["RainfallLog1p"] = np.log1p(df["Rainfall"].clip(lower=0))

    df["HasRainfall"] = np.where(
        df["Rainfall"].isna(),
        np.nan,
        (df["Rainfall"] > 0).astype(int),
    )

    df["HighHumidity3pm"] = np.where(
        df["Humidity3pm"].isna(),
        np.nan,
        (df["Humidity3pm"] >= 70).astype(int),
    )

    df["StrongWindGust"] = np.where(
        df["WindGustSpeed"].isna(),
        np.nan,
        (df["WindGustSpeed"] >= 50).astype(int),
    )

    return df


# ============================================================
# Gestion des colonnes fortement manquantes
# ============================================================

def drop_high_missing_columns(
    df: pd.DataFrame,
    high_missing_columns: List[str],
    protected_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Supprime les colonnes fortement incomplètes si le scénario est activé.

    Par défaut, on conserve ces colonnes.
    Cette fonction sert uniquement à tester un scénario alternatif.
    """
    df = df.copy()

    if protected_columns is None:
        protected_columns = [TARGET, "Date"]

    columns_to_drop = [
        col for col in high_missing_columns
        if col in df.columns and col not in protected_columns
    ]

    return df.drop(columns=columns_to_drop)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes inutiles après feature engineering.

    Colonnes supprimées si présentes :
    - Date         : extraite en variables temporelles (Year, Month, DayOfYear, …)
    - date_import  : colonne technique Airflow, hors features modèle
    - data_source  : colonne technique Airflow, hors features modèle
    - run_id       : colonne technique Airflow, hors features modèle
    """
    df = df.copy()

    columns_to_drop = ["Date"] + TECHNICAL_COLUMNS
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    return df.drop(columns=existing_columns)


# ============================================================
# Séparation X / y
# ============================================================

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare les variables explicatives X et la cible y.

    Les colonnes techniques (TECHNICAL_COLUMNS) sont exclues de X
    si elles n'ont pas déjà été supprimées par drop_unused_columns.
    """
    cols_to_exclude = [TARGET] + [
        c for c in TECHNICAL_COLUMNS if c in df.columns
    ]
    X = df.drop(columns=cols_to_exclude)
    y = df[TARGET]

    return X, y


# ============================================================
# Identification des types de variables
# ============================================================

def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifie automatiquement les variables numériques et catégorielles."""
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


# ============================================================
# Préprocesseur scikit-learn
# ============================================================

def make_one_hot_encoder():
    """
    Crée un OneHotEncoder compatible avec plusieurs versions de scikit-learn.

    L'import est effectué localement pour permettre de tester les étapes
    pandas même si scikit-learn / SciPy pose problème dans l'environnement.
    """
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
):
    """
    Construit le préprocesseur complet (non entraîné).

    Tous les imports sklearn sont effectués localement pour permettre de
    tester les étapes pandas même si scikit-learn / SciPy pose problème.

    Variables numériques :
    - IterativeImputer : modélise chaque variable manquante à partir de toutes
      les autres via régression bayésienne, préservant les corrélations naturelles
      (ex : Temp9am / MaxTemp) — évite le biais d'une valeur unique ;
    - indicateurs de valeurs manquantes (add_indicator=True) ;
    - RobustScaler : utilise médiane + IQR, insensible aux outliers de Rainfall
      et WindGustSpeed contrairement à StandardScaler.

    Variable Location :
    - imputation par la modalité la plus fréquente ;
    - TargetEncoder si disponible (scikit-learn >= 1.3) : encode chaque ville
      par sa probabilité de pluie historique, réduisant 49 colonnes OHE à 1.
      Fallback vers OneHotEncoder sinon.

    Autres variables catégorielles :
    - imputation par la modalité la plus fréquente ;
    - encodage One-Hot.

    Le préprocesseur doit être intégré dans un Pipeline avec le modèle
    et fitté uniquement sur X_train.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                IterativeImputer(
                    max_iter=10,
                    random_state=42,
                    add_indicator=True,
                ),
            ),
            ("scaler", RobustScaler()),
        ]
    )

    # TargetEncoder disponible depuis scikit-learn 1.3
    try:
        from sklearn.preprocessing import TargetEncoder
        location_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", TargetEncoder(target_type="binary")),
            ]
        )
        logger.info("Encodage Location : TargetEncoder")
    except ImportError:
        location_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_one_hot_encoder()),
            ]
        )
        logger.warning(
            "TargetEncoder indisponible (scikit-learn < 1.3). "
            "Fallback vers OneHotEncoder pour Location."
        )

    other_cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    location_features = [col for col in categorical_features if col == "Location"]
    other_cat_features = [col for col in categorical_features if col != "Location"]

    transformers: List = [("numeric", numeric_pipeline, numeric_features)]

    if location_features:
        transformers.append(("location", location_pipeline, location_features))

    if other_cat_features:
        transformers.append(("categorical", other_cat_pipeline, other_cat_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ============================================================
# Préparation déterministe
# ============================================================

def prepare_dataframe(
    df: pd.DataFrame,
    high_missing_columns: Optional[List[str]] = None,
    drop_high_missing: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Applique les transformations déterministes.

    Cette fonction ne fait pas :
    - d'imputation statistique ;
    - de standardisation ;
    - d'encodage One-Hot ou target encoding.

    Ces opérations restent dans le ColumnTransformer pour éviter la fuite de données.

    Prévention de la fuite de données (data leakage)
    -------------------------------------------------
    Seules les transformations « sans mémoire » sont appliquées ici :
    encodage binaire de la cible, parsing de date, feature engineering
    arithmétique. Toute transformation qui apprend une statistique sur
    les données (moyenne, médiane, probabilités de pluie par ville…)
    est déléguée au ColumnTransformer, qui ne sera fitté que sur X_train
    à l'intérieur d'un Pipeline sklearn. Ainsi le test set reste un
    témoin strictement indépendant.
    """
    validate_schema(df)

    df = clean_target(df)
    df = encode_rain_today(df)
    df = parse_date_column(df)

    df_with_date = df.copy()

    df = add_temporal_features(df)
    df = add_cyclical_features(df)
    df = encode_wind_directions(df)
    df = add_weather_features(df)

    if drop_high_missing and high_missing_columns is not None:
        df = drop_high_missing_columns(
            df=df,
            high_missing_columns=high_missing_columns,
            protected_columns=[TARGET, "Date"],
        )

    df = drop_unused_columns(df)

    X, y = split_features_target(df)

    return X, y, df_with_date


# ============================================================
# Split train/test
# ============================================================

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    df_for_sort: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "random",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare les données en train/test.

    L'import sklearn est effectué localement pour permettre de tester les
    étapes pandas même si scikit-learn / SciPy pose problème.

    Stratégies disponibles :
    - random    : split aléatoire stratifié — acceptable pour un benchmark rapide.
    - temporal  : split chronologique (toutes les dates d'entraînement précèdent
                  celles du test) — obligatoire en production pour simuler la
                  prédiction sur des jours futurs et éviter le look-ahead leakage :
                  avec un split aléatoire, des observations d'un même lieu et
                  d'une même période peuvent se retrouver simultanément dans le
                  train ET le test, gonflant artificiellement les métriques.
    """
    if split_strategy not in ["random", "temporal"]:
        raise ValueError("split_strategy doit être 'random' ou 'temporal'.")

    if split_strategy == "random":
        from sklearn.model_selection import train_test_split
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    if df_for_sort is None or "Date" not in df_for_sort.columns:
        raise ValueError(
            "Pour split_strategy='temporal', df_for_sort doit contenir Date."
        )

    ordered_index = df_for_sort.sort_values("Date").index

    X_ordered = X.loc[ordered_index]
    y_ordered = y.loc[ordered_index]

    split_position = int(len(X_ordered) * (1 - test_size))

    X_train = X_ordered.iloc[:split_position]
    X_test = X_ordered.iloc[split_position:]
    y_train = y_ordered.iloc[:split_position]
    y_test = y_ordered.iloc[split_position:]

    return X_train, X_test, y_train, y_test


# ============================================================
# Fonction principale
# ============================================================

def prepare_data(
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "random",
    missing_threshold: float = HIGH_MISSING_THRESHOLD,
    drop_high_missing: bool = False,
    save_report: bool = True,
    source: str = "csv",
    connection_uri: Optional[str] = None,
    table_name: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict[str, object]:
    """
    Prépare les données pour la modélisation.

    Supporte trois sources : CSV (défaut, rétrocompatible), PostgreSQL, API REST.

    Par défaut :
    - les colonnes avec plus de 30 % de valeurs manquantes sont conservées ;
    - elles sont traitées par IterativeImputer + indicateur de missingness ;
    - la suppression est disponible comme scénario alternatif (drop_high_missing=True).

    Les colonnes techniques (date_import, data_source, run_id) provenant de
    PostgreSQL sont exclues des features du modèle mais disponibles dans les
    données brutes pour le monitoring Airflow.

    Parameters
    ----------
    data_path : str, optional
        Chemin CSV. Utilisé si source="csv". Défaut : DEFAULT_DATA_PATH.
    source : {"csv", "postgres", "api"}
        Source de données. Par défaut : "csv".
    connection_uri : str, optional
        URI SQLAlchemy. Requis si source="postgres".
    table_name : str, optional
        Nom de la table. Requis si source="postgres".
    api_url : str, optional
        URL de l'endpoint. Requis si source="api".
    """
    raw_df = load_dataset(
        source=source,
        data_path=data_path,
        connection_uri=connection_uri,
        table_name=table_name,
        api_url=api_url,
    )

    validate_schema(raw_df)

    missing_report = build_missing_values_report(raw_df)

    high_missing_columns = identify_high_missing_columns(
        df=raw_df,
        threshold=missing_threshold,
        exclude_columns=[TARGET],
    )

    if save_report:
        save_missing_values_report(missing_report)

    X, y, df_with_date = prepare_dataframe(
        df=raw_df,
        high_missing_columns=high_missing_columns,
        drop_high_missing=drop_high_missing,
    )

    numeric_features, categorical_features = identify_feature_types(X)

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X=X,
        y=y,
        df_for_sort=df_with_date,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_report": missing_report,
        "high_missing_columns": high_missing_columns,
    }


# ============================================================
# Tests locaux
# ============================================================

def check_preprocessor_works(data: Dict[str, object]) -> None:
    """
    Vérifie que le préprocesseur fonctionne correctement.

    Anti-leakage : fit_transform sur X_train uniquement, puis transform sur X_test.
    IterativeImputer apprend les distributions sur le train ; TargetEncoder calcule
    les probabilités de pluie par ville sur le train ; RobustScaler calcule médiane
    et IQR sur le train. Aucune statistique du test set ne remonte vers le train.

    En production, sauvegarder le preprocessor fitté (joblib.dump) comme artefact
    versionné afin que l'inférence applique exactement les mêmes transformations
    sans re-fitter — évite la dérive de transformation entre entraînement et serving.
    """
    preprocessor = data["preprocessor"]

    X_train_transformed = preprocessor.fit_transform(
        data["X_train"], data["y_train"]
    )
    X_test_transformed = preprocessor.transform(data["X_test"])

    logger.info("Test du préprocesseur : OK")
    logger.info("X_train transformé : %s", X_train_transformed.shape)
    logger.info("X_test transformé  : %s", X_test_transformed.shape)


def display_preprocessing_summary(data: Dict[str, object]) -> None:
    """Affiche un résumé du prétraitement."""
    logger.info("Prétraitement terminé.")
    logger.info("-" * 70)

    logger.info("X_train : %s", data["X_train"].shape)
    logger.info("X_test  : %s", data["X_test"].shape)
    logger.info("y_train : %s", data["y_train"].shape)
    logger.info("y_test  : %s", data["y_test"].shape)

    logger.info(
        "Distribution de y_train :\n%s",
        data["y_train"].value_counts(normalize=True).round(4),
    )
    logger.info(
        "Distribution de y_test :\n%s",
        data["y_test"].value_counts(normalize=True).round(4),
    )

    logger.info(
        "Colonnes avec plus de 30 %% de valeurs manquantes : %s",
        data["high_missing_columns"],
    )

    logger.info("Nombre de variables numériques : %d", len(data["numeric_features"]))
    logger.info("%s", data["numeric_features"])

    logger.info(
        "Nombre de variables catégorielles : %d",
        len(data["categorical_features"]),
    )
    logger.info("%s", data["categorical_features"])

    logger.info(
        "Top 10 des variables avec valeurs manquantes :\n%s",
        data["missing_report"].head(10),
    )

    logger.info("Préprocesseur :\n%s", data["preprocessor"])


def parse_arguments() -> argparse.Namespace:
    """Gère les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Prétraitement du dataset WeatherAUS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="csv",
        choices=["csv", "postgres", "api"],
        help="Source de données : csv (défaut), postgres ou api.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Chemin vers weatherAUS.csv (source=csv). "
            f"Défaut : {DEFAULT_DATA_PATH}"
        ),
    )

    parser.add_argument(
        "--connection-uri",
        type=str,
        default=None,
        help=(
            "URI de connexion PostgreSQL (source=postgres). "
            "Exemple : postgresql://USER:PWD@localhost:5432/DB_NAME"
        ),
    )

    parser.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="Nom de la table PostgreSQL (source=postgres).",
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=(
            "URL de l'endpoint API (source=api). "
            "Exemple : http://localhost:8057/meteo"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du jeu de test.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Graine de reproductibilité.",
    )

    parser.add_argument(
        "--split-strategy",
        type=str,
        default="random",
        choices=["random", "temporal"],
        help="Stratégie de split : random ou temporal.",
    )

    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=HIGH_MISSING_THRESHOLD,
        help="Seuil de valeurs manquantes. Exemple : 0.30 pour 30 %%.",
    )

    parser.add_argument(
        "--drop-high-missing",
        action="store_true",
        help="Supprime les colonnes avec trop de valeurs manquantes.",
    )

    parser.add_argument(
        "--no-save-report",
        action="store_true",
        help="Désactive la sauvegarde du rapport des valeurs manquantes.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    args = parse_arguments()

    preprocessing_data = prepare_data(
        source=args.source,
        data_path=args.data_path,
        connection_uri=args.connection_uri,
        table_name=args.table_name,
        api_url=args.api_url,
        test_size=args.test_size,
        random_state=args.random_state,
        split_strategy=args.split_strategy,
        missing_threshold=args.missing_threshold,
        drop_high_missing=args.drop_high_missing,
        save_report=not args.no_save_report,
    )

    display_preprocessing_summary(preprocessing_data)
    check_preprocessor_works(preprocessing_data)
