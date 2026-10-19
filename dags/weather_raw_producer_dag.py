#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Orchestration Airflow d'ingestion des données

    Description :
        DAG orchestrant la simulation d'arrivée des données brutes
        
        Chaîne orchestrée :
        
            WeatherRawDataProducer
                    |
                    v
            Génération des fichiers CSV dans dossier RAW
                    |
                    v
            WeatherLoader
                    |
                    v
            Chargement PostgreSQL (raw_weather)

        Le pipeline est exécuté dans un environnement conteneurisé Docker
        et permet de simuler des arrivées progressives de données afin de
        reproduire un contexte industriel d'ingestion batch.

    Version :
        1.0.0

    Historique :
        2026-07-11  -  Création du module
===============================================================================
"""
import os

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from core.settings import SETTINGS

import logging

# Logger Airflow
logger = logging.getLogger("airflow.task")

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
        source=SETTINGS["docker"]["host_logs_dir"],
        target=SETTINGS["docker"]["container_logs_target"],
        type="bind",
    )
]


# -----------------------------
# CALLBACK ERREUR
# -----------------------------
def log_simulation_failure(context):
    """
    Callback exécuté en cas d'échec d'une tâche de simulation.
    Loggue l'erreur dans les logs Airflow.
    """

    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    run_id = context.get("run_id")
    exception = str(context.get("exception"))
    try_number = context.get("task_instance").try_number

    logger.error(
        {
            "event": "simulation_failed",
            "message": "Échec de la tâche de simulation",
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
    "on_failure_callback": log_simulation_failure
}


 
# -----------------------------
# DÉFINITION DU DAG
# -----------------------------
with DAG(
    dag_id="weather_rawdata_producer",
    default_args=default_args,
    description="Simulation d'arrivée des données brutes",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops26", "weather", "rawdata-simulation"],
) as dag:

    logger.info(
        {
            "event": "dag_start",
            "message": "DAG weather_rawdata_producer initialisé",
            "schedule": "none",
        }
    )
    

    # -----------------------------------------
    # Etape 1 - Simulation d'arrivé de données brutes
    # Génère des fichiers csv dans dossier raw
    # -----------------------------------------
    generate_raw_batches = DockerOperator(
        task_id="generate_raw_batches",
        image=SETTINGS["docker"]["data-integration_image"],
        command=(
            "{% set c = var.json.get('WEATHER_RAW_PRODUCER_CONFIG', {}) %}"
            "python /app/weather_raw_producer_entrypoint.py"
            " --batch-size {{ c.get('batch_size', 500) }}"
            " --max-batches {{ c.get('max_batches', 5) }}"
            " --delay {{ c.get('delay', 5) }}"
            "{% if c.get('start_batch') %} --start-batch {{ c.get('start_batch') }}{% endif %}"
        ),
        pool="producer_pool",
        environment={
            "AIRFLOW_CONTAINER_DATA_DIR": os.environ["AIRFLOW_CONTAINER_DATA_DIR"],
            "AIRFLOW_CONTAINER_LOGS_DIR": os.environ["AIRFLOW_CONTAINER_LOGS_DIR"],
            "WEATHER_HOST_DATA_DIR": os.environ["WEATHER_HOST_DATA_DIR"],
            "WEATHER_HOST_LOGS_DIR": os.environ["WEATHER_HOST_LOGS_DIR"],
            "WEATHER_HOST_MODELS_DIR":os.environ["WEATHER_HOST_MODELS_DIR"], 
            "WEATHER_HOST_METRICS_DIR":os.environ["WEATHER_HOST_METRICS_DIR"],
        },
        docker_url="unix://var/run/docker.sock",
        network_mode="weather",
        mounts=mounts,
        # Limites CPU & RAM
        mem_limit="2g",
        cpus=1,
        auto_remove="force",
        mount_tmp_dir=False,
        do_xcom_push=False,
    )
    