#!/usr/bin/env python3

"""
Description :
    Recupere la derniere version du modele enregistree par evaluate_model
    dans le Model Registry MLflow, compare son F1 a celui du modele en
    production, et la promeut en production seulement si elle est meilleure
"""

import os

import mlflow
from mlflow.tracking import MlflowClient


from core.settings import SETTINGS
from core.logger import get_logger


logger = get_logger("promote_model")

MODEL_NAME = SETTINGS["models"]["registered_model_name"]


def get_new_version(client):
    # evaluate_model enregistre chaque run comme une nouvelle version sans stage
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    if not versions:
        return None
    return versions[0]


def get_production_version(client):
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        return None
    return versions[0]


def f1_of(client, version):
    run = client.get_run(version.run_id)
    return float(run.data.metrics.get("f1", 0.0))


def main():
    # L'adresse DagsHub et les identifiants sont lus dans l'environnement (.env)
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    new = get_new_version(client)
    if new is None:
        print("Aucune nouvelle version a promouvoir")
        return

    new_f1 = f1_of(client, new)
    prod = get_production_version(client)
    prod_f1 = f1_of(client, prod) if prod is not None else None

    # Promotion seulement si le nouveau modele est meilleur
    # ou s'il n'y a pas encore de modele en production
    if prod_f1 is None or new_f1 > prod_f1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Version {new.version} promue en production. F1 {new_f1}")
    else:
        print(f"Version {new.version} non promue. F1 {new_f1} contre {prod_f1} en production")


if __name__ == "__main__":
    main() 
