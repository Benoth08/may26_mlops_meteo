#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Orchestration Airflow de l'entraînement

    Description :
        DAG qui orchestre la chaîne complète en conteneurs Docker :
        ingestion, préparation, recherche des hyperparamètres, entraînement,
        évaluation et promotion du modèle.

    Version :
        1.0.0

    Historique :
        2026-07-11  -  Création du module
===============================================================================
"""

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import datetime, timedelta
from docker.types import Mount

import logging

# Logger Airflow
logger = logging.getLogger("airflow.task")


# -----------------------------
# MOUNTS DOCKER
# -----------------------------
# Montages utilises par la tache d'ingestion
mounts = [
    Mount(
        source="/home/ubuntu/scripts/projet_weather/src/data",
        target="/data",
        type="bind",
    ),
    Mount(
        source="/home/ubuntu/scripts/projet_weather/logs",
        target="/logs",
        type="bind",
    )
]

# Montages utilises par les taches ML
# Les taches se passent des fichiers entre elles : le dataset, le modele et les
# scores sont stockes sur l'hote pour rester disponibles d'une tache a l'autre
mounts_ml = [
    Mount(
        source="/home/ubuntu/scripts/projet_weather/data",
        target="/app/data",
        type="bind",
    ),
    Mount(
        source="/home/ubuntu/scripts/projet_weather/models",
        target="/app/models",
        type="bind",
    ),
    Mount(
        source="/home/ubuntu/scripts/projet_weather/metrics",
        target="/app/metrics",
        type="bind",
    ),
]


# -----------------------------
# CALLBACK ERREUR
# -----------------------------
def log_task_failure(context):
    # Appele quand une tache echoue, pour tracer l'erreur dans les logs Airflow
    logger.error(
        {
            "event": "task_failed",
            "dag_id": context.get("dag").dag_id,
            "task_id": context.get("task_instance").task_id,
            "run_id": context.get("run_id"),
            "exception": str(context.get("exception")),
        }
    )


# -----------------------------
# CONFIGURATION PAR DEFAUT DAG
# -----------------------------
default_args = {
    "owner": "weather",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": log_task_failure,
}


# -----------------------------
# DEFINITION DU DAG
# -----------------------------
with DAG(
    dag_id="weather_training_pipeline",
    default_args=default_args,
    description="Pipeline complete : ingestion, preparation, entrainement, evaluation, promotion",
    start_date=datetime(2026, 6, 1),
    schedule=None,
    catchup=False,
    tags=["mlops26", "weather", "training"],
) as dag:

    # -----------------------------------------
    # Etape 1 - Ingestion des donnees
    # Charge le CSV dans la base PostgreSQL
    # -----------------------------------------
    ingestion = DockerOperator(
        task_id="load_weather_csv",
        image="data-integration:latest",
        command="python entrypoint.py",
        environment={
            "POSTGRES_WTH_DB": "{{ var.value.POSTGRES_WTH_DB }}",
            "POSTGRES_WTH_USER": "{{ var.value.POSTGRES_WTH_USER }}",
            "POSTGRES_WTH_PASSWORD": "{{ var.value.POSTGRES_WTH_PASSWORD }}",
        },
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        mounts=mounts,
        auto_remove="force",
        mount_tmp_dir=False,
    )

    # -----------------------------------------
    # Etape 2 - Preparation des donnees
    # Pretraitement et decoupage chronologique train / test
    # -----------------------------------------
    preparation = DockerOperator(
        task_id="prepare_data",
        image="data-preparation:latest",
        command="python -m src.data.make_dataset",
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        mounts=mounts_ml,
        auto_remove="force",
        mount_tmp_dir=False,
    )

    # -----------------------------------------
    # Etape 3 - Recherche des hyperparametres
    # GridSearch avec validation croisee temporelle
    # -----------------------------------------
    grid_search = DockerOperator(
        task_id="grid_search",
        image="model-training:latest",
        command="python -m src.models.grid_search",
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        mounts=mounts_ml,
        auto_remove="force",
        mount_tmp_dir=False,
    )

    # -----------------------------------------
    # Etape 4 - Entrainement du modele final
    # Entraine le pipeline avec les meilleurs hyperparametres
    # -----------------------------------------
    training = DockerOperator(
        task_id="train_model",
        image="model-training:latest",
        command="python -m src.models.train_model",
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        mounts=mounts_ml,
        auto_remove="force",
        mount_tmp_dir=False,
    )

    # -----------------------------------------
    # Etape 5 - Evaluation du modele
    # Calcule les metriques sur le jeu de test et enregistre le modele dans MLflow
    # Les variables MLflow permettent la connexion au serveur DagsHub
    # -----------------------------------------
    evaluation = DockerOperator(
        task_id="evaluate_model",
        image="model-evaluation:latest",
        command="python -m src.models.evaluate_model",
        environment={
            "MLFLOW_TRACKING_URI": "{{ var.value.MLFLOW_TRACKING_URI }}",
            "MLFLOW_TRACKING_USERNAME": "{{ var.value.MLFLOW_TRACKING_USERNAME }}",
            "MLFLOW_TRACKING_PASSWORD": "{{ var.value.MLFLOW_TRACKING_PASSWORD }}",
        },
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        mounts=mounts_ml,
        auto_remove="force",
        mount_tmp_dir=False,
    )

    # -----------------------------------------
    # Etape 6 - Promotion du modele
    # Compare le F1 du nouveau modele a celui en production.
    # Le nouveau modele n'est promu que s'il est meilleur.
    # -----------------------------------------
    promotion = DockerOperator(
        task_id="promote_model",
        image="model-validation:latest",
        command="python -m src.models.promote_model",
        environment={
            "MLFLOW_TRACKING_URI": "{{ var.value.MLFLOW_TRACKING_URI }}",
            "MLFLOW_TRACKING_USERNAME": "{{ var.value.MLFLOW_TRACKING_USERNAME }}",
            "MLFLOW_TRACKING_PASSWORD": "{{ var.value.MLFLOW_TRACKING_PASSWORD }}",
        },
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        mounts=mounts_ml,
        auto_remove="force",
        mount_tmp_dir=False,
    )

    # -----------------------------
    # ENCHAINEMENT DES TACHES
    # -----------------------------
    # Chaque tache attend que la precedente ait reussi
    ingestion >> preparation >> grid_search >> training >> evaluation >> promotion
