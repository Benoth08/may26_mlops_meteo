#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Orchestration Airflow de l'entraînement

    Description :
        DAG qui orchestre la chaîne suivante en conteneurs Docker :
        recherche des hyperparamètres, entraînement, évaluation et promotion du modèle.

    Version :
        1.0.0

    Historique :
        2026-07-11  -  Création du module
===============================================================================
"""
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from datetime import datetime, timedelta
from docker.types import Mount

import logging

# Logger Airflow
logger = logging.getLogger("airflow.task.models")


# -----------------------------
# MOUNTS DOCKER
# -----------------------------
mounts = [
    Mount(
        source="/home/ubuntu/projet_weather/data",
        target="/data",
        type="bind",
    ),
    Mount(
        source="/home/ubuntu/projet_weather/models",
        target="/models",
        type="bind",
    ),
    Mount(
        source="/home/ubuntu/projet_weather/metrics",
        target="/app/metrics",
        type="bind",
    ),
    Mount(
        source="/home/ubuntu/projet_weather/logs",
        target="/logs",
        type="bind",
    )
]


# -----------------------------
# CALLBACK ERREUR
# -----------------------------
def log_models_failure(context):
    """
    Callback exécuté en cas d'échec d'une tâche de models.
    Loggue l'erreur dans les logs Airflow.
    """

    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    run_id = context.get("run_id")
    exception = str(context.get("exception"))
    try_number = context.get("task_instance").try_number

    logger.error(
        {
            "event": "models_failed",
            "message": "Échec de la tâche de models",
            "dag_id": dag_id,
            "task_id": task_id,
            "run_id": run_id,
            "try_number": try_number,
            "exception": exception,
        }
    )


# -----------------------------
# CONFIGURATION PAR DÉFAUT DAG
# -----------------------------
default_args = {
    "owner": "weather",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": log_models_failure
}


# -----------------------------
# DÉFINITION DU DAG
# -----------------------------
with DAG(
    dag_id="weather_models",
    default_args=default_args,
    description="Entrainement du model : GridSearch → Train final model → Evaluate",
    start_date=datetime(2026, 6, 1),
    schedule=None,
    catchup=False,
    tags=["mlops26", "weather", "model", "model-training", "model-evaluating"],
) as dag:

    logger.info(
        {
            "event": "dag_start",
            "message": "DAG weather_models initialisé",
            "schedule": "0 * * * *",
        }
    )

    # -----------------------------------------
    # Etape 3.1 - Recherche des hyperparametres
    # GridSearch avec validation croisee temporelle
    # -----------------------------------------
    grid_search = DockerOperator(
        task_id="grid_search",
        image="models:latest",
        command="python models/grid_search.py",
        pool="ml_pool",
        mounts=mounts,
        # Limites CPU & RAM
        mem_limit="4g",
        cpus=2,
        environment={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        auto_remove="force",
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )
    
    logger.info(
        {
            "event": "task_registered",
            "task_id": "grid_search",
            "message": "Etape 1/4 terminée."
        }
    )
    
    # -----------------------------------------
    # Etape 3.2 - Entrainement du modele final
    # Entraine le pipeline avec les meilleurs hyperparametres
    # -----------------------------------------
    train_model = DockerOperator(
        task_id="train_model",
        image="models:latest",
        command="python models/train_model.py",
        pool="ml_pool",
        mounts=mounts,
        # Limites CPU & RAM
        mem_limit="4g",
        cpus=2,
        environment={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        auto_remove="force",
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )

    logger.info(
        {
            "event": "task_registered",
            "task_id": "train_model",
            "message": "Etape 2/4 terminée."
        }
    )
    
    # -----------------------------------------
    # Etape 3.3 - Evaluation du modele
    # Calcule les metriques sur le jeu de test et enregistre le modele dans MLflow
    # Les variables MLflow permettent la connexion au serveur DagsHub
    # -----------------------------------------
    evaluate_model = DockerOperator(
        task_id="evaluate_model",
        image="models:latest",
        command="python models/evaluate_model.py",
        pool="ml_pool",
        mounts=mounts,
        # Limites CPU & RAM
        mem_limit="4g",
        cpus=2,
        environment={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MLFLOW_TRACKING_URI": "{{ var.value.MLFLOW_TRACKING_URI }}",
            "MLFLOW_TRACKING_USERNAME": "{{ var.value.MLFLOW_TRACKING_USERNAME }}",
            "MLFLOW_TRACKING_PASSWORD": "{{ var.value.MLFLOW_TRACKING_PASSWORD }}",
        },
        auto_remove="force",
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )

    logger.info(
        {
            "event": "task_registered",
            "task_id": "evaluate_model",
            "message": "Etape 3/4 terminée."
        }
    )
    
    # -----------------------------------------
    # Etape 3.4 - Promotion du modele
    # Compare le F1 du nouveau modele a celui en production.
    # Le nouveau modele n'est promu que s'il est meilleur.
    # -----------------------------------------
    promotion_model = DockerOperator(
        task_id="promote_model",
        image="models:latest",
        command="python models/promote_model.py",
        pool="ml_pool",
        mounts=mounts,
        # Limites CPU & RAM
        mem_limit="4g",
        cpus=2,
        environment={
            "MLFLOW_TRACKING_URI": "{{ var.value.MLFLOW_TRACKING_URI }}",
            "MLFLOW_TRACKING_USERNAME": "{{ var.value.MLFLOW_TRACKING_USERNAME }}",
            "MLFLOW_TRACKING_PASSWORD": "{{ var.value.MLFLOW_TRACKING_PASSWORD }}",
        },
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        auto_remove="force",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )
    
    logger.info(
        {
            "event": "task_registered",
            "task_id": "promote_model",
            "message": "Etape 4/4 terminée."
        }
    )
    
    # -----------------------------------------
    # Orchestration
    # -----------------------------------------
    grid_search >> train_model >> evaluate_model >> promotion_model

