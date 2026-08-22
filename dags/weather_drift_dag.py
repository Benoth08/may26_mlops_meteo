#!/usr/bin/env python3
"""
===============================================================================
    Surveillance de la dérive des données
    DAG hebdomadaire comparant les données météorologiques récentes avec celles observées
    à la même période il y a un an.
    Il donne deux informations distinctes, la dérive des données et leur qualité.
    La tâche d’alerte échoue quand l’une des deux se déclenche, afin de la rendre visible.
    Le réentraînement se déclenchera automatiquement dans un second temps,
    une fois les seuils calés sur des mesures réelles.
===============================================================================
"""
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import datetime, timedelta
from docker.types import Mount

import logging

# Logger Airflow
logger = logging.getLogger("airflow.task.drift")


# Le script lit les données dans Postgres
mounts = [
    Mount(
        source="/home/ubuntu/projet_weather/reports",
        target="/reports",
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
def log_drift_failure(context):
    """
    Callback exécuté en cas d'échec d'une tâche de surveillance.
    Loggue l'erreur dans les logs Airflow.
    """

    logger.error(
        {
            "event": "drift_failed",
            "message": "Échec de la tâche de surveillance de dérive",
            "dag_id": context.get("dag").dag_id,
            "task_id": context.get("task_instance").task_id,
            "run_id": context.get("run_id"),
            "exception": str(context.get("exception")),
        }
    )


# Contrôle du résultat
def controler_surveillance(**context):
    # La tache precedente affiche sa decision en derniere ligne.
    # Airflow la recupere via XCom. Trois valeurs possibles :
    #   OK       : rien a signaler
    #   DERIVE   : les donnees ont change, un reentrainement se justifie
    #   QUALITE  : les donnees sont abimees, il faut corriger le pipeline
    #              avant de reentrainer quoi que ce soit
    decision = context["ti"].xcom_pull(task_ids="detect_drift")

    if decision is None:
        raise AirflowException("Aucun resultat recupere de la detection")

    decision = str(decision).strip()

    if decision == "QUALITE":
        logger.error(
            {
                "event": "qualite_degradee",
                "message": "Qualite des donnees degradee, ne pas reentrainer",
            }
        )
        raise AirflowException(
            "Qualite des donnees degradee. Verifier l ingestion avant de "
            "reentrainer. Voir le rapport dans le dossier reports.")

    if decision == "DERIVE":
        logger.error(
            {
                "event": "derive_detectee",
                "message": "Derive des donnees detectee, reentrainement conseille",
            }
        )
        # L'echec de la tache rend l'alerte visible dans Airflow
        raise AirflowException(
            "Derive detectee. Voir le rapport dans le dossier reports.")

    logger.info({"event": "rien_a_signaler", "decision": decision})
    print("Pas de derive ni de probleme de qualite")


# Config
default_args = {
    "owner": "weather",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": log_drift_failure,
}


# Définition
with DAG(
    dag_id="weather_drift",
    default_args=default_args,
    description="Surveillance hebdomadaire de la derive des donnees",
    start_date=datetime(2026, 6, 1),
    # Tous les lundis a 4h du matin, soit plus souvent que le reentrainement
    schedule="0 4 * * 1",
    catchup=False,
    tags=["mlops26", "weather", "monitoring", "drift"],
) as dag:

    # Etape 1 - Detection de la derive
    # Compare la fenetre recente a la meme periode un an plus tot.
    # La comparaison annuelle evite de confondre saison et derive.
    detect_drift = DockerOperator(
        task_id="detect_drift",
        image="monitoring:latest",
        command="python monitoring/drift_detection.py",
        pool="weather_pool",
        mounts=mounts,
        mem_limit="4g",
        cpus=2,
        environment={
            "POSTGRES_WTH_DB": "{{ var.value.POSTGRES_WTH_DB }}",
            "POSTGRES_WTH_USER": "{{ var.value.POSTGRES_WTH_USER }}",
            "POSTGRES_WTH_PASSWORD": "{{ var.value.POSTGRES_WTH_PASSWORD }}",
            "POSTGRES_WTH_HOST": "{{ var.value.POSTGRES_WTH_HOST }}",
            "POSTGRES_WTH_PORT": "{{ var.value.POSTGRES_WTH_PORT }}",
            # Taille des deux fenetres comparees, en jours
            "DRIFT_WINDOW_DAYS": "{{ var.value.DRIFT_WINDOW_DAYS }}",
            # Part de colonnes derivees declenchant l'alerte
            "DRIFT_SHARE_THRESHOLD": "{{ var.value.DRIFT_SHARE_THRESHOLD }}",
            # Hausse du taux de valeurs manquantes declenchant l'alerte qualite
            "QUALITY_MISSING_INCREASE": "{{ var.value.QUALITY_MISSING_INCREASE }}",
        },
        auto_remove="force",
        docker_url="unix:///var/run/docker.sock",
        network_mode="weather",
        mount_tmp_dir=False,
        do_xcom_push=True,
    )


    # Etape 2 - Alerte
    # Lit la decision de l'etape precedente et echoue si une derive ou un
    # probleme de qualite est detecte, pour que le signal soit visible.
    alerte_surveillance = PythonOperator(
        task_id="alerte_surveillance",
        python_callable=controler_surveillance,
    )

    # Orchestration
    detect_drift >> alerte_surveillance
