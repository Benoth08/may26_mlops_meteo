#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Orchestration Airflow d des données

    Description :
        DAG qui orchestre la chaîne suivante en conteneurs Docker :
        ingestion, préparation des données

    Version :
        1.0.0

    Historique :
        2026-07-11  -  Création du module
===============================================================================
"""
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import datetime, timedelta
from docker.types import Mount

from core.settings import SETTINGS

import logging

# Logger Airflow
logger = logging.getLogger("airflow.task.preprocessing")


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
        source=SETTINGS["docker"]["host_logs_dir"],
        target=SETTINGS["docker"]["container_logs_target"],
        type="bind",
    )
]

# Identifiants rendus par Jinja au moment de l'exécution, jamais au parsing :
# ils ne sont ni sérialisés avec le DAG, ni relus en base à chaque parsing.
# export_dataset.py se rabat sur load_postgres_config() sans POSTGRES_URI.
POSTGRES_ENV = {
    "POSTGRES_WTH_HOST": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).host }}",
    "POSTGRES_WTH_PORT": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).port }}",
    "POSTGRES_WTH_DB": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).schema }}",
    "POSTGRES_WTH_USER": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).login }}",
    "POSTGRES_WTH_PASSWORD": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).password }}",
}

COMMON_ENV = {
    "AIRFLOW_CONTAINER_DATA_DIR": os.environ["AIRFLOW_CONTAINER_DATA_DIR"],
    "AIRFLOW_CONTAINER_LOGS_DIR": os.environ["AIRFLOW_CONTAINER_LOGS_DIR"],
    "WEATHER_HOST_DATA_DIR": os.environ["WEATHER_HOST_DATA_DIR"],
    "WEATHER_HOST_LOGS_DIR": os.environ["WEATHER_HOST_LOGS_DIR"],
    "WEATHER_HOST_MODELS_DIR": os.environ["WEATHER_HOST_MODELS_DIR"],
    "WEATHER_HOST_METRICS_DIR": os.environ["WEATHER_HOST_METRICS_DIR"],
}

# -----------------------------
# CALLBACK ERREUR
# -----------------------------
def log_preprocessing_failure(context):
    """
    Callback exécuté en cas d'échec d'une tâche de preprocessing.
    Loggue l'erreur dans les logs Airflow.
    """

    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    run_id = context.get("run_id")
    exception = str(context.get("exception"))
    try_number = context.get("task_instance").try_number

    logger.error(
        {
            "event": "preprocessing_failed",
            "message": "Échec de la tâche de preprocessing",
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": log_preprocessing_failure
}


# -----------------------------
# DÉFINITION DU DAG
# -----------------------------
with DAG(
    dag_id="weather_preprocessing",
    default_args=default_args,
    description="Préparation des données météo en vue d'entrainer le modele de prévision",
    start_date=datetime(2026, 7, 5),
    schedule="*/20 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlops26", "weather", "data-preprocessing"],
) as dag:

    logger.info(
        {
            "event": "dag_start",
            "message": "DAG weather_preprocessing initialisé",
            "schedule": "*/20 * * * *",
        }
    )
    
    # -----------------------------------------
    # Etape 2.1 - Preparation des donnees
    # Pretraitement et decoupage chronologique train / test
    # -----------------------------------------
    extracting_weather_dataset = DockerOperator(
        task_id="extracting_weather_dataset",
        image=SETTINGS["docker"]["data-preprocessing_image"],
        command="python data_preprocessing/export_dataset.py",
        pool="weather_pool",
        # Aucune donnée extraite (0 ligne) => le script sort en code 99.
        # Airflow marque alors la tâche "skipped" au lieu de "failed", ce qui
        # fait aussi passer les tâches en aval en "skipped" (trigger_rule
        # par défaut = all_success) : pas de données, pas de traitement.
        skip_on_exit_code=99,
        # Limites CPU & RAM
        mem_limit="2g",
        cpus=1,
        environment={**POSTGRES_ENV, **COMMON_ENV},
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mounts=mounts,
        auto_remove="force",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )
    
    logger.info(
        {
            "event": "task_registered",
            "task_id": "extracting_weather_dataset",
            "message": "Etape 1/3"
        }
    )

    # -----------------------------------------
    # Etape 2.2 - Preparation des donnees
    # Pretraitement et decoupage chronologique train / test
    # -----------------------------------------
    preprocessing_weather_data = DockerOperator(
        task_id="preprocessing_weather_data",
        image=SETTINGS["docker"]["data-preprocessing_image"],
        command="python data_preprocessing/make_dataset.py",
        pool="weather_pool",
        # Filet de sécurité : si la tâche s'exécute malgré tout sur un
        # dataset brut vide, le script sort en code 99 => skip plutôt que fail.
        skip_on_exit_code=99,
        # Limites CPU & RAM
        mem_limit="2g",
        cpus=1,
        environment={**POSTGRES_ENV, **COMMON_ENV},
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mounts=mounts,
        auto_remove="force",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )

    logger.info(
        {
            "event": "task_registered",
            "task_id": "preprocessing_weather_data",
            "message": "Etape 2/3"
        }
    )

    # -----------------------------------------
    # Etape 2.3 - Normalisation des données
    # 
    # -----------------------------------------
    normalizing_weather_data = DockerOperator(
        task_id="normalizing_weather_data",
        image=SETTINGS["docker"]["data-preprocessing_image"],
        command="python data_preprocessing/normalize_data.py",
        pool="weather_pool",
        mounts=mounts,
        # Filet de sécurité : si la tâche s'exécute malgré tout sans données
        # en amont, le script sort en code 99 => skip plutôt que fail.
        skip_on_exit_code=99,
        # Limites CPU & RAM
        mem_limit="2g",
        cpus=1,
        environment={**POSTGRES_ENV, **COMMON_ENV},
        auto_remove="force",
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )

    logger.info(
        {
            "event": "task_registered",
            "task_id": "normalizing_weather_data",
            "message": "Etape 3/3"
        }
    )
    
    extracting_weather_dataset >> preprocessing_weather_data >> normalizing_weather_data
