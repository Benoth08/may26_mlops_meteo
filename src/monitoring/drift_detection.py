#!/usr/bin/env python3
"""
===============================================================================
    Data Drift :
        On compare les données actuelles avec celles de la même période un an
        en arrière. On a fait ceci car la saisonnalité pose un problème,
        l'écart entre l'été et l'hiver est 
        signalé comme une dérive à chaque exécution.

        Deux signaux utilisés :
        - la dérive des données, qui justifie un réentraînement
        - la qualité des données, qui signale plutôt un problème de pipeline

        Le script écrit un rapport HTML et affiche sa décision en dernière
        ligne, pour qu'Airflow puisse la récupérer.
===============================================================================
"""

import os
import sys

import pandas as pd
from sqlalchemy import create_engine

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

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
]

# Fenêtres d'analyse
FENETRE_JOURS = int(os.environ.get("DRIFT_WINDOW_DAYS", "30"))

# Seuil de dérive (à adapter).
# hasard statistique. Un seuil trop bas donnerait des alertes permanentes.
SEUIL_PART_DERIVEE = float(os.environ.get("DRIFT_SHARE_THRESHOLD", "0.30"))

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

    # Supression des colonnes techniques et la date
    colonnes = [c for c in courant.columns if c not in COLONNES_EXCLUES]
    courant = courant[colonnes]
    reference = reference[colonnes]

    rapport = Report([DataDriftPreset(), DataSummaryPreset()])
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

    logger.info({
        "event": "surveillance_analysee",
        "colonnes_derivees": nombre_derive,
        "part_derivee": part_derive,
        "seuil_derive": SEUIL_PART_DERIVEE,
        "derive_detectee": derive_detectee,
        "lignes_dupliquees": doublons,
        "part_manquants_courant": manquants_courant,
        "part_manquants_reference": manquants_reference,
        "hausse_manquants": hausse_manquants,
        "qualite_degradee": qualite_degradee,
        "rapport": REPORT_PATH if rapport_ecrit else None,
    })

    print("Colonnes derivees : {} sur {} analysees".format(
        int(nombre_derive), len(colonnes)))
    print("Part derivee : {:.2f} pour un seuil de {:.2f}".format(
        part_derive, SEUIL_PART_DERIVEE))
    print("Lignes dupliquees : {}".format(int(doublons)))
    print("Valeurs manquantes : {:.2f} contre {:.2f} en reference".format(
        manquants_courant, manquants_reference))

    # Décisions
    if qualite_degradee:
        decision = "QUALITE"
    elif derive_detectee:
        decision = "DERIVE"
    else:
        decision = "OK"

    # Faire en sorte que Airflow récupère l'information
    print(decision)


if __name__ == "__main__":
    main()
