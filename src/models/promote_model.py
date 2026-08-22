#!/usr/bin/env python3

"""
    Récupère la dernière version du modèle par `evaluate_model` dans le Model Registry MLflow,
    compare le F1 à celui du modèle en production, et ne le mets en production que si le score est meilleur.
    La comparaison est effectuée sur le même jeu de test pour les deux modèles.
    Le modèle de production est rechargé depuis le Registry et réévalué sur le jeu de test courant, 
    au lieu de réutiliser son F1 d’origine qui avait été calculé sur un jeu de test plus ancien.
    Un seuil minimal de F1 déclenche une alerte lorsque le modèle choisi est trop faible en valeur absolue.
"""

import os
import sys

import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

from core.settings import SETTINGS
from core.logger import get_logger


logger = get_logger("promote_model")

REGISTERED_MODEL_NAME = SETTINGS["models"]["registered_model_name"]

# Jeu de test courant
DATASET_PATH = SETTINGS["paths"]["processed"] / SETTINGS["models"]["dataset"]

# Seuil minimal de F1 attendu pour le modèle en production.
# En dessous, on alerte
F1_ALERT_THRESHOLD = float(os.environ.get("F1_ALERT_THRESHOLD", "0.55"))


def get_new_version(client):
    # evaluate_model enregistre chaque run comme une nouvelle version
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
    if not versions:
        return None
    return versions[0]


def get_production_version(client):
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        return None
    return versions[0]


def load_test_set():
    # On lit le même dataset que celui utilisé par evaluate_model
    data = joblib.load(DATASET_PATH)
    return data["X_test"], data["y_test"]


def f1_on_test_set(version, X_test, y_test):
    # Recharge le modele depuis le Registry et calcule son F1 sur le jeu de test fourni
    uri = "models:/{}/{}".format(REGISTERED_MODEL_NAME, version.version)
    model = mlflow.sklearn.load_model(uri)
    y_pred = model.predict(X_test)
    return float(f1_score(y_test, y_pred))


def promote(client, version):
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,
    )


def alerte_si_sous_seuil(f1_retenu, contexte):
    # Alerte quand le modèle en production est sous le seuil minimal.
    # Airflow est mis en echec si alerte et qu'elle soit visible.
    if f1_retenu < F1_ALERT_THRESHOLD:
        message = (
            "ALERTE : F1 de {:.4f} sous le seuil minimal de {:.4f} ({})".format(
                f1_retenu, F1_ALERT_THRESHOLD, contexte)
        )
        logger.error({
            "event": "f1_sous_seuil",
            "f1": f1_retenu,
            "seuil": F1_ALERT_THRESHOLD,
            "contexte": contexte,
        })
        print(message)
        sys.exit(1)


def main():
    # L'adresse DagsHub et les identifiants sont lus dans l'environnement (.env)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    new = get_new_version(client)
    if new is None:
        print("Aucune nouvelle version a promouvoir")
        return

    try:
        X_test, y_test = load_test_set()
    except FileNotFoundError:
        logger.error({"event": "dataset_introuvable", "chemin": str(DATASET_PATH)})
        print("Jeu de test introuvable, promotion annulee")
        sys.exit(1)

    new_f1 = f1_on_test_set(new, X_test, y_test)
    prod = get_production_version(client)

    # Premier passage : aucun modèle en production
    if prod is None:
        promote(client, new)
        print("Version {} envoyée en production (premier modele). F1 {:.4f}".format(
            new.version, new_f1))
        alerte_si_sous_seuil(new_f1, "premier modele promu")
        return
    try:
        prod_f1 = f1_on_test_set(prod, X_test, y_test)
    except Exception as e:
        # Si échec on ne fait rien
        logger.error({"event": "modele_production_illisible", "error": str(e)}, exc_info=True)
        print("Modele de production illisible, promotion annulee par prudence")
        sys.exit(1)

    if new_f1 > prod_f1:
        promote(client, new)
        print("Version {} promue en production. F1 {:.4f} contre {:.4f}".format(
            new.version, new_f1, prod_f1))
        alerte_si_sous_seuil(new_f1, "nouveau modele promu")
    else:
        print("Version {} non promue. F1 {:.4f} contre {:.4f} en production".format(
            new.version, new_f1, prod_f1))
        alerte_si_sous_seuil(prod_f1, "modele de production conserve")


if __name__ == "__main__":
    main()
