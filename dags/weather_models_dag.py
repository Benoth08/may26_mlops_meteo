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
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from datetime import datetime, timedelta
from docker.types import Mount

from core.settings import SETTINGS

import logging

# Logger Airflow
logger = logging.getLogger("airflow.task.models")


# -----------------------------
# MOUNTS DOCKER
# -----------------------------
mounts = [
    Mount(
        source=SETTINGS["docker"]["host_data_dir"],
        target=SETTINGS["docker"]["container_data_target"],
        type="bind",
    ),
    Mount(
        source=SETTINGS["docker"]["host_models_dir"],
        target=SETTINGS["docker"]["container_models_target"],
        type="bind",
    ),
    Mount(
        source=SETTINGS["docker"]["host_metrics_dir"],
        target=SETTINGS["docker"]["container_metrics_target"],
        type="bind",
    ),
    Mount(
        source=SETTINGS["docker"]["host_logs_dir"],
        target=SETTINGS["docker"]["container_logs_target"],
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
    # Déclenchable à la fois par weather_training_trigger et manuellement
    # depuis l'UI : sans cette limite, deux runs concurrents écriraient en
    # même temps sur les mêmes fichiers (candidate_model.joblib,
    # best_params.joblib, puis model.joblib côté promote_model.py).
    max_active_runs=1,
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
        image=SETTINGS["docker"]["models_image"],
        command="python models/grid_search.py",
        pool="ml_pool",
        mounts=mounts,
        # Pas de dataset prétraité (preprocessing "skipped" faute de données)
        # => le script sort en code 99 => skip plutôt que fail, en cascade
        # sur train_model / evaluate_model / promote_model.
        skip_on_exit_code=99,
        # Limites CPU & RAM
        mem_limit="4g",
        cpus=2,
        environment={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "AIRFLOW_CONTAINER_DATA_DIR": os.environ["AIRFLOW_CONTAINER_DATA_DIR"],
            "AIRFLOW_CONTAINER_LOGS_DIR": os.environ["AIRFLOW_CONTAINER_LOGS_DIR"],
            "WEATHER_HOST_DATA_DIR": os.environ["WEATHER_HOST_DATA_DIR"],
            "WEATHER_HOST_LOGS_DIR": os.environ["WEATHER_HOST_LOGS_DIR"],
            "WEATHER_HOST_MODELS_DIR":os.environ["WEATHER_HOST_MODELS_DIR"],
            "WEATHER_HOST_METRICS_DIR":os.environ["WEATHER_HOST_METRICS_DIR"],
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
        image=SETTINGS["docker"]["models_image"],
        command="python models/train_model.py",
        pool="ml_pool",
        mounts=mounts,
        # Filet de sécurité : artefact manquant => skip plutôt que fail.
        skip_on_exit_code=99,
        # Limites CPU & RAM
        mem_limit="4g",
        cpus=2,
        environment={
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            # Contexte transmis par weather_training_trigger (conf du
            # TriggerDagRunOperator). Absent en cas de déclenchement manuel
            # du DAG, d'où les valeurs par défaut.
            "TRAINING_REASON": "{{ (dag_run.conf or {}).get('training_reason', 'manual') }}",
            "TRAINING_NEW_ROWS": "{{ (dag_run.conf or {}).get('new_rows', 0) }}",
            "TRAINING_ROW_COUNT": "{{ (dag_run.conf or {}).get('current_row_count', 0) }}",
            "AIRFLOW_CONTAINER_DATA_DIR": os.environ["AIRFLOW_CONTAINER_DATA_DIR"],
            "AIRFLOW_CONTAINER_LOGS_DIR": os.environ["AIRFLOW_CONTAINER_LOGS_DIR"],
            "WEATHER_HOST_DATA_DIR": os.environ["WEATHER_HOST_DATA_DIR"],
            "WEATHER_HOST_LOGS_DIR": os.environ["WEATHER_HOST_LOGS_DIR"],
            "WEATHER_HOST_MODELS_DIR": os.environ["WEATHER_HOST_MODELS_DIR"],
            "WEATHER_HOST_METRICS_DIR": os.environ["WEATHER_HOST_METRICS_DIR"],
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
        image=SETTINGS["docker"]["models_image"],
        command="python models/evaluate_model.py",
        pool="ml_pool",
        mounts=mounts,
        # Filet de sécurité : artefact manquant => skip plutôt que fail.
        skip_on_exit_code=99,
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
            "AIRFLOW_CONTAINER_DATA_DIR": os.environ["AIRFLOW_CONTAINER_DATA_DIR"],
            "AIRFLOW_CONTAINER_LOGS_DIR": os.environ["AIRFLOW_CONTAINER_LOGS_DIR"],
            "WEATHER_HOST_DATA_DIR": os.environ["WEATHER_HOST_DATA_DIR"],
            "WEATHER_HOST_LOGS_DIR": os.environ["WEATHER_HOST_LOGS_DIR"],
            "WEATHER_HOST_MODELS_DIR": os.environ["WEATHER_HOST_MODELS_DIR"],
            "WEATHER_HOST_METRICS_DIR": os.environ["WEATHER_HOST_METRICS_DIR"],
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
        image=SETTINGS["docker"]["models_image"],
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
            # Requis pour TrainingStateRepository (model_training_status).
            "POSTGRES_WTH_HOST": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).host }}",
            "POSTGRES_WTH_PORT": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).port }}",
            "POSTGRES_WTH_DB": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).schema }}",
            "POSTGRES_WTH_USER": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).login }}",
            "POSTGRES_WTH_PASSWORD": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).password }}",
            # Contexte transmis par weather_training_trigger (conf du
            # TriggerDagRunOperator). Absent en cas de déclenchement manuel
            # du DAG, d'où les valeurs par défaut.
            "TRAINING_REASON": "{{ (dag_run.conf or {}).get('training_reason', 'manual') }}",
            "TRAINING_ROW_COUNT": "{{ (dag_run.conf or {}).get('current_row_count', 0) }}",
            "AIRFLOW_CONTAINER_DATA_DIR": os.environ["AIRFLOW_CONTAINER_DATA_DIR"],
            "AIRFLOW_CONTAINER_LOGS_DIR": os.environ["AIRFLOW_CONTAINER_LOGS_DIR"],
            "WEATHER_HOST_DATA_DIR": os.environ["WEATHER_HOST_DATA_DIR"],
            "WEATHER_HOST_LOGS_DIR": os.environ["WEATHER_HOST_LOGS_DIR"],
            "WEATHER_HOST_MODELS_DIR":os.environ["WEATHER_HOST_MODELS_DIR"], 
            "WEATHER_HOST_METRICS_DIR":os.environ["WEATHER_HOST_METRICS_DIR"],
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

