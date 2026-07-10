#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Promotion du modèle

    Description :
        Compare le F1 du dernier modèle entraîné à celui du modèle en production
        et promeut le nouveau modèle s'il est meilleur

    Version :
        1.0.0

    Historique :
        2026-07-10  -  Création du module
===============================================================================
"""

import os

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "weather-rain-model"
EXPERIMENT_NAME = "weather-rain-prediction"


def main():
    # Connexion au serveur MLflow
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    # On recupere le dernier entrainement et son F1
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["start_time DESC"],
        max_results=1,
    )
    last_run = runs[0]
    new_f1 = float(last_run.data.metrics["f1"])

    # On enregistre le nouveau modele dans le registre
    result = mlflow.register_model(f"runs:/{last_run.info.run_id}/model", MODEL_NAME)

    # On cherche le modele actuellement en production et son F1
    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    prod_f1 = None
    if prod:
        prod_run = client.get_run(prod[0].run_id)
        prod_f1 = float(prod_run.data.metrics.get("f1", 0.0))

    # Promotion seulement si le nouveau modele est meilleur
    if prod_f1 is None or new_f1 > prod_f1:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Modele promu en production. F1 {new_f1}")
    else:
        print(f"Modele non promu. F1 {new_f1} contre {prod_f1} en production")


if __name__ == "__main__":
    main()
