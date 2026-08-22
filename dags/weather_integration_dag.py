#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Orchestration Airflow d'ingestion des données

    Description :
        DAG qui orchestre la chaîne suivante en conteneurs Docker :
        ingestion

    Version :
        1.0.0

    Historique :
        2026-07-11  -  Création du module
===============================================================================
"""

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from datetime import datetime, timedelta
from docker.types import Mount

import logging

# Logger Airflow
logger = logging.getLogger("airflow.task")


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
        source="/home/ubuntu/projet_weather/logs",
        target="/logs",
        type="bind",
    )
]


# -----------------------------
# CALLBACK ERREUR
# -----------------------------
def log_integration_failure(context):
    """
    Callback exécuté en cas d'échec d'une tâche d'intégration.
    Loggue l'erreur dans les logs Airflow.
    """

    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    run_id = context.get("run_id")
    exception = str(context.get("exception"))
    try_number = context.get("task_instance").try_number

    logger.error(
        {
            "event": "integration_failed",
            "message": "Échec de la tâche d'intégration",
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
    "on_failure_callback": log_integration_failure
}


# -----------------------------
# DÉFINITION DU DAG
# -----------------------------
with DAG(
    dag_id="weather_csv_integration",
    default_args=default_args,
    description="Intégration quotidienne des données météo",
    start_date=datetime(2026, 6, 1),
    schedule=None,
    catchup=False,
    tags=["mlops26", "weather", "data-integration"],
) as dag:

    logger.info(
        {
            "event": "dag_start",
            "message": "DAG weather_csv_integration initialisé",
            "schedule": "0 * * * *",
        }
    )

    # -----------------------------------------
    # Etape 1 - Ingestion des donnees
    # Charge le CSV dans la base PostgreSQL
    # -----------------------------------------
    integration_weather_data = DockerOperator(
        task_id="load_weather_csv",
        image="data-integration:latest",
        command="python entrypoint.py",
        pool="weather_pool",
        environment={
            "POSTGRES_WTH_HOST": "{{ var.value.POSTGRES_WTH_HOST }}",
            "POSTGRES_WTH_PORT": "{{ var.value.POSTGRES_WTH_PORT }}",
            "POSTGRES_WTH_DB": "{{ var.value.POSTGRES_WTH_DB }}",
            "POSTGRES_WTH_USER": "{{ var.value.POSTGRES_WTH_USER }}",
            "POSTGRES_WTH_PASSWORD": "{{ var.value.POSTGRES_WTH_PASSWORD }}",
        },
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mounts=mounts,
        # Limites CPU & RAM
        mem_limit="2g",
        cpus=1,
        auto_remove="force",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )

    logger.info(
        {
            "event": "task_registered",
            "task_id": "load_weather_csv",
            "message": "Tache terminée."
        }
    )

    integration_weather_data
