#!/usr/bin/env python3
"""
===============================================================================
    Data Drift :
        On compare les données actuelles avec celles de la même période un an
        en arrière. On a fait ceci car la saisonnalité pose un problème,
        l'écart entre l'été et l'hiver est 
        signalé comme une dérive à chaque exécution.

        Trois signaux utilisés :
        - la dérive des données, qui justifie un réentraînement
        - la qualité des données, qui signale plutôt un problème de pipeline
        - la dérive des prédictions, c'est-à-dire ce que sort le modèle en
          service. C'est le signal le plus précoce : il ne demande pas de
          savoir s'il a vraiment plu, donc il alerte avant que la
          performance ne se dégrade.

        Le script écrit un rapport HTML et affiche sa décision en dernière
        ligne, pour qu'Airflow puisse la récupérer.
===============================================================================
"""

import os
import sys

import pandas as pd
from sqlalchemy import create_engine

import joblib

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from build_features import (
    convert_types, encode_rain_today, parse_date_column,
    add_temporal_features, add_cyclical_features,
    encode_wind_directions, add_weather_features, drop_unused_columns,
)

from core.settings import SETTINGS
from core.config import load_postgres_config, ConfigError
from core.logger import get_logger


logger = get_logger("drift_detection")

TABLE_RAW = SETTINGS["postgres"]["table_raw"]

# Dossier monté depuis l'hôte par le DAG
REPORT_PATH = "/reports/drift_report.html"

# Colonnes techniques ajoutées par l'ingestion. Ces colonnes changent à chaque ingestion donc on les supprimes
COLONNES_EXCLUES = [
    SETTINGS["postgres"]["importdate_column_norm"],
    SETTINGS["postgres"]["importrunid_column_norm"],
    SETTINGS["postgres"]["importsource_column_norm"],
    "date",
    "id",
]

# Fenêtres d'analyse
FENETRE_JOURS = int(os.environ.get("DRIFT_WINDOW_DAYS", "30"))

# Seuil de distance au dela duquel UNE colonne est declaree derivee.
# La valeur par defaut d'Evidently, 0.10, est trop sensible pour ces donnees :
# en coupant une meme periode en deux au hasard, on obtenait deja 50 a 65
# pour cent de colonnes "derivees", c'est-a-dire du pur bruit.
# A 0.30, ce bruit tombe a zero alors que les vraies differences subsistent.
SEUIL_COLONNE = float(os.environ.get("DRIFT_COLUMN_THRESHOLD", "0.30"))

# Part de colonnes derivees a partir de laquelle on declare une derive.
# Calibre avec le seuil ci-dessus, sur les donnees reelles :
#   donnees homogenes   -> 0.00 a 0.05
#   annees consecutives -> 0.09 a 0.14
SEUIL_PART_DERIVEE = float(os.environ.get("DRIFT_SHARE_THRESHOLD", "0.10"))

# Nom normalise de la cible, retiree avant de predire
TARGET = SETTINGS["target"]["column_norm"]

# Modele servi par l'API, dans le dossier partage monte par le DAG.
# On surveille ce que predit le modele reellement en service.
MODEL_PATH = "/models/model.joblib"

# Hausse du taux de valeurs manquantes
# Seuil à adapter
SEUIL_HAUSSE_MANQUANTS = float(os.environ.get("QUALITY_MISSING_INCREASE", "0.10"))

# Nombre minimal de lignes dans chaque fenêtre pour que la comparaison ait un sens
MIN_LIGNES = 30

def lire_fenetre(engine, date_debut, date_fin):
    requete = (
        'SELECT * FROM {} WHERE "date" > \'{}\' AND "date" <= \'{}\''.format(
            TABLE_RAW, date_debut.date(), date_fin.date())
    )
    return pd.read_sql(requete, engine)


def indicateur(resultat, prefixe):
    # Retrouve un indicateur Evidently a partir du debut de son nom
    for mesure in resultat.dict()["metrics"]:
        if mesure["metric_name"].startswith(prefixe):
            return mesure["value"]
    return None


def part_manquants(df):
    # Part de cellules vides dans le tableau
    if len(df) == 0 or len(df.columns) == 0:
        return 0.0
    return float(df.isna().sum().sum()) / (len(df) * len(df.columns))


def charger_modele():
    # Meme logique de chargement que l'API : l'artefact peut etre un
    # dictionnaire contenant le pipeline, ou le pipeline directement.
    artefact = joblib.load(MODEL_PATH)
    if isinstance(artefact, dict):
        return artefact.get("pipeline")
    return artefact


def featuriser(df):
    # Rejoue la meme chaine de preparation que l'entrainement et que l'API.
    # La cible est retiree : en inference on ne la connait pas.
    df = df.copy()
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    df = convert_types(df)
    if df.empty:
        return df
    df = encode_rain_today(df)
    df = parse_date_column(df)
    df = add_temporal_features(df)
    df = add_cyclical_features(df)
    df = encode_wind_directions(df)
    df = add_weather_features(df)
    df = drop_unused_columns(df)
    return df


def derive_prediction(courant_brut, reference_brut):
    # Compare ce que le modele predit sur les deux fenetres.
    # Renvoie None si le signal n'est pas calculable : le modele peut etre
    # absent au premier demarrage, cela ne doit pas bloquer le reste.
    if not os.path.exists(MODEL_PATH):
        logger.info({"event": "modele_absent", "chemin": MODEL_PATH})
        return None

    modele = charger_modele()
    if modele is None:
        logger.error({"event": "modele_illisible", "chemin": MODEL_PATH})
        return None

    xc = featuriser(courant_brut)
    xr = featuriser(reference_brut)
    if xc.empty or xr.empty:
        logger.error({"event": "featurisation_vide"})
        return None

    pc = modele.predict_proba(xc)[:, 1]
    pr = modele.predict_proba(xr)[:, 1]

    # Meme methode et meme seuil que pour la derive des donnees,
    # appliques a la seule colonne des probabilites predites.
    preset = DataDriftPreset(
        num_method="wasserstein",
        num_threshold=SEUIL_COLONNE,
    )
    res = Report([preset]).run(
        current_data=pd.DataFrame({"probabilite": pc}),
        reference_data=pd.DataFrame({"probabilite": pr}),
    )

    distance = None
    for mesure in res.dict()["metrics"]:
        if "column=probabilite" in mesure["metric_name"]:
            distance = float(mesure["value"])
            break

    if distance is None:
        logger.error({"event": "distance_prediction_introuvable"})
        return None

    return {
        "distance": distance,
        "derive": distance >= SEUIL_COLONNE,
        "proba_moyenne_courant": float(pc.mean()),
        "proba_moyenne_reference": float(pr.mean()),
        "taux_pluie_courant": float((pc >= 0.5).mean()),
        "taux_pluie_reference": float((pr >= 0.5).mean()),
    }


def main():
    try:
        cfg = load_postgres_config()
    except ConfigError as e:
        logger.error({"event": "config_error", "error": str(e)})
        sys.exit(1)

    engine = create_engine(cfg.sqlalchemy_uri)

    # Date la plus récente présente en base
    date_max = pd.read_sql(
        'SELECT MAX("date") AS derniere FROM {}'.format(TABLE_RAW), engine
    )["derniere"].iloc[0]

    if date_max is None:
        logger.error({"event": "table_vide", "table": TABLE_RAW})
        print("Table vide, aucune analyse possible")
        sys.exit(1)

    date_max = pd.Timestamp(date_max)

    # Fenêtre courante : les derniers jours disponibles
    debut_courant = date_max - pd.Timedelta(days=FENETRE_JOURS)

    # Fenêtre de référence : la même période, un an plus tôt
    fin_reference = date_max - pd.Timedelta(days=365)
    debut_reference = fin_reference - pd.Timedelta(days=FENETRE_JOURS)

    courant = lire_fenetre(engine, debut_courant, date_max)
    reference = lire_fenetre(engine, debut_reference, fin_reference)

    logger.info({
        "event": "fenetres_lues",
        "courant": [str(debut_courant.date()), str(date_max.date())],
        "reference": [str(debut_reference.date()), str(fin_reference.date())],
        "lignes_courant": len(courant),
        "lignes_reference": len(reference),
    })

    # Sans assez d'historique, la comparaison annuelle n'est pas possible
    if len(reference) < MIN_LIGNES or len(courant) < MIN_LIGNES:
        logger.error({
            "event": "historique_insuffisant",
            "lignes_courant": len(courant),
            "lignes_reference": len(reference),
            "minimum": MIN_LIGNES,
        })
        print("Historique insuffisant pour comparer, analyse annulee")
        sys.exit(1)

    # On garde une copie brute : le modele a besoin de toutes les colonnes
    # d'origine pour rejouer la preparation, y compris la date.
    courant_brut = courant.copy()
    reference_brut = reference.copy()

    # Supression des colonnes techniques et la date
    colonnes = [c for c in courant.columns if c not in COLONNES_EXCLUES]
    courant = courant[colonnes]
    reference = reference[colonnes]

    # La methode statistique est figee volontairement. Par defaut Evidently
    # change de test en dessous de 1000 lignes, ce qui ferait varier le
    # comportement de la surveillance selon le volume de donnees.
    derive_preset = DataDriftPreset(
        num_method="wasserstein",
        cat_method="jensenshannon",
        num_threshold=SEUIL_COLONNE,
        cat_threshold=SEUIL_COLONNE,
    )
    rapport = Report([derive_preset, DataSummaryPreset()])
    resultat = rapport.run(current_data=courant, reference_data=reference)

    rapport_ecrit = True
    try:
        resultat.save_html(REPORT_PATH)
    except Exception as e:
        rapport_ecrit = False
        logger.error({"event": "rapport_non_ecrit", "error": str(e)})


    # Premier signal: dérive des données
    valeur_derive = indicateur(resultat, "DriftedColumnsCount")

    if valeur_derive is None:
        logger.error({"event": "indicateur_derive_introuvable"})
        print("Resultat Evidently illisible")
        sys.exit(1)

    nombre_derive = valeur_derive["count"]
    part_derive = valeur_derive["share"]
    derive_detectee = part_derive >= SEUIL_PART_DERIVEE

    # Second signal: qualité des données
    valeur_doublons = indicateur(resultat, "DuplicatedRowCount")
    doublons = float(valeur_doublons) if valeur_doublons is not None else 0.0
    manquants_courant = part_manquants(courant)
    manquants_reference = part_manquants(reference)
    hausse_manquants = manquants_courant - manquants_reference
    qualite_degradee = (
        doublons > 0 or hausse_manquants >= SEUIL_HAUSSE_MANQUANTS
    )

    # Troisieme signal: derive des predictions
    prediction = derive_prediction(courant_brut, reference_brut)
    derive_prediction_detectee = bool(prediction and prediction["derive"])

    logger.info({
        "event": "surveillance_analysee",
        "colonnes_derivees": nombre_derive,
        "part_derivee": part_derive,
        "seuil_colonne": SEUIL_COLONNE,
        "seuil_derive": SEUIL_PART_DERIVEE,
        "derive_detectee": derive_detectee,
        "lignes_dupliquees": doublons,
        "part_manquants_courant": manquants_courant,
        "part_manquants_reference": manquants_reference,
        "hausse_manquants": hausse_manquants,
        "qualite_degradee": qualite_degradee,
        "prediction": prediction,
        "derive_prediction_detectee": derive_prediction_detectee,
        "rapport": REPORT_PATH if rapport_ecrit else None,
    })

    print("Colonnes derivees : {} sur {} analysees".format(
        int(nombre_derive), len(colonnes)))
    print("Part derivee : {:.2f} pour un seuil de {:.2f}".format(
        part_derive, SEUIL_PART_DERIVEE))
    print("Lignes dupliquees : {}".format(int(doublons)))
    print("Valeurs manquantes : {:.2f} contre {:.2f} en reference".format(
        manquants_courant, manquants_reference))

    if prediction is None:
        print("Derive des predictions : non calculee (modele indisponible)")
    else:
        print("Pluie predite : {:.1f} pour cent contre {:.1f} en reference".format(
            100 * prediction["taux_pluie_courant"],
            100 * prediction["taux_pluie_reference"]))
        print("Derive des predictions : {:.2f} pour un seuil de {:.2f}".format(
            prediction["distance"], SEUIL_COLONNE))

    # Décisions
    # La qualite prime : reentrainer sur des donnees abimees les apprendrait
    # telles quelles. Une derive des donnees ou des predictions justifie en
    # revanche un reentrainement.
    if qualite_degradee:
        decision = "QUALITE"
    elif derive_detectee or derive_prediction_detectee:
        decision = "DERIVE"
    else:
        decision = "OK"

    # Faire en sorte que Airflow récupère l'information
    print(decision)


if __name__ == "__main__":
    main()
