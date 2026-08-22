#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather : Predict next-day rain in Australia
    ---------------------------------------------------------------------------
    Sujet :
        Orchestration Airflow d'ingestion des données

    Description :
        DAG qui attend l'arrivée de fichiers CSV dans RAW puis lance l'intégration
        PostgreSQL.

    Sources possibles :
        - WeatherRawDataProducer
        - Dépôt manuel de fichiers CSV

    Fonctionnement :
        - Surveillance du répertoire RAW
        - Détection des fichiers weather*.csv sans distinction de casse
        - Verrouillage des fichiers par renommage en .processing
        - Intégration dans PostgreSQL via Docker
        - En cas d'échec définitif (retries épuisés) : mise en quarantaine
          du fichier plutôt que remise en RAW, pour éviter une boucle
          infinie de re-traitement sur un fichier corrompu
        - Si d'autres fichiers sont disponibles :
              attente non bloquante d'un délai anti-surcharge (TimeDeltaSensor)
              puis déclenchement immédiat d'un nouveau run
        - Si aucun fichier ne reste :
              fin du run
              puis reprise de la surveillance au prochain créneau
              du schedule (1 minute)
    Version :
        1.1.0

    Historique :
        2026-07-11  -  Création du module
        2026-08-04  -  Passage en mode détection de fichiers
        2026-08-12  -  Correctifs :
                         - branchement manquant de wait_before_retrigger
                         - suppression du time.sleep() bloquant (+ import
                           manquant) dans more_files_pending
                         - import TimeDeltaSensor remonté en tête de fichier
                         - mise en quarantaine des fichiers en échec définitif
                           au lieu d'une remise en RAW (évite une boucle
                           infinie de re-traitement)
===============================================================================
"""
import os

from airflow import DAG
from airflow.providers.standard.sensors.python import PythonSensor
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.python import ShortCircuitOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.standard.sensors.time_delta import TimeDeltaSensor
from airflow.providers.docker.operators.docker import DockerOperator

from datetime import datetime, timedelta
from docker.types import Mount
from pathlib import Path

from core.logger import SETTINGS # pyright: ignore[reportMissingImports]
from core.settings import SETTINGS # pyright: ignore[reportMissingImports]

# Logger Airflow
import logging

logger = logging.getLogger(__name__)

##logger = get_logger("weather_integration", console=True)

RAW_DIRECTORY = Path(SETTINGS["docker"]["airflow_data_dir"]) / SETTINGS["docker"]["folder_data_raw"]
NETWORK_MODE = SETTINGS["docker"]["network"]

# Répertoire de quarantaine pour les fichiers dont l'intégration a
# définitivement échoué (retries épuisés). Ces fichiers ne sont plus
# jamais repris automatiquement par get_raw_csv_files(), ce qui évite
# de boucler indéfiniment sur un CSV corrompu.
#
# Si un dossier dédié est défini dans la configuration
# (SETTINGS["docker"]["folder_data_quarantine"]), on l'utilise ;
# sinon on retombe sur un sous-dossier "quarantine" de RAW_DIRECTORY.
QUARANTINE_DIRECTORY = (
    Path(SETTINGS["docker"]["airflow_data_dir"]) / SETTINGS["docker"]["folder_data_quarantine"]
    if "folder_data_quarantine" in SETTINGS["docker"]
    else RAW_DIRECTORY / "quarantine"
)

# Délai entre deux relances lorsque des fichiers restent à traiter.
#
# Exemple :
#     10  -> attente de 10 secondes
#     30  -> attente de 30 secondes
#
# L'objectif est d'éviter d'enchaîner les runs trop rapidement
# lorsque WeatherRawDataProducer dépose plusieurs fichiers.
RETRIGGER_POKE_INTERVAL_SECONDS = 30
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


def rollback_processing_file(context):
    """
    Callback exécuté en cas d'échec DÉFINITIF d'intégration (Airflow
    n'appelle on_failure_callback qu'une fois tous les retries épuisés ;
    les échecs intermédiaires déclenchent on_retry_callback, pas celui-ci).

    Auparavant, ce callback renommait le fichier ".processing" pour lui
    rendre son nom d'origine dans RAW_DIRECTORY. Comme le fichier
    redevenait alors "disponible" pour get_raw_csv_files(), il était
    aussitôt repris par le run suivant (déclenché par trigger_next_run) et
    échouait de nouveau à l'identique : boucle infinie sur un fichier
    corrompu, qui bloquait le traitement des fichiers suivants.

    Comportement corrigé : le fichier est déplacé vers QUARANTINE_DIRECTORY
    plutôt que remis dans RAW_DIRECTORY. Il n'est alors plus jamais repris
    automatiquement et reste disponible pour une inspection manuelle.

    XCom ne contient que le NOM du fichier ".processing" (ex :
    "weather_batch_001.csv.processing") ; on le recombine ici avec
    RAW_DIRECTORY, le répertoire tel que vu côté Airflow.

    Exemple :
        weather_batch_001.csv.processing  (dans RAW_DIRECTORY)
                    ↓
        weather_batch_001.csv.processing  (dans QUARANTINE_DIRECTORY)
    """

    # On loggue systématiquement l'échec : ce callback remplace celui de
    # default_args pour cette tâche, il doit donc en reprendre le rôle.
    log_integration_failure(context)

    task_instance = context["task_instance"]

    # Récupération du nom du fichier traité depuis XCom
    processing_filename = task_instance.xcom_pull(
        task_ids="find_raw_file"
    )

    if not processing_filename:
        logger.warning(
            {
                "event": "rollback_skipped",
                "message": "Aucun fichier processing trouvé dans XCom"
            }
        )
        return

    processing_path = RAW_DIRECTORY / processing_filename

    if not processing_path.exists():

        logger.warning(
            {
                "event": "rollback_file_missing",
                "file": str(processing_path)
            }
        )

        return

    try:

        QUARANTINE_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True
        )
     

        quarantine_path = QUARANTINE_DIRECTORY / processing_path.name

        processing_path.rename(
            quarantine_path
        )

        logger.error(
            {
                "event": "file_quarantined",
                "message": "Échec définitif de l'intégration, fichier mis en quarantaine",
                "processing_file": str(processing_path),
                "quarantine_file": str(quarantine_path)
            }
        )

    except Exception as e:

        logger.error(
            {
                "event": "quarantine_failed",
                "processing_file": str(processing_path),
                "exception": str(e)
            }
        )

        raise


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



# ==========================================================
# Détection arrivée fichiers RAW
# ==========================================================

def get_raw_csv_files():
    """
    Retourne les fichiers CSV RAW dont le nom commence par weather
    sans tenir compte de la casse.
    Exemples acceptés :
        weather_batch_001.csv
        Weather_batch_002.csv
        WEATHER_IMPORT.csv

    Exclus :
        config_xxx.json
        fichier_test.csv
        weather_xxx.csv.processing
    """

    files = []

    if not RAW_DIRECTORY.exists():
        logger.warning(
            {
                "event": "raw_directory_missing",
                "folder": str(RAW_DIRECTORY),
            }
        )
        return files
    for file in RAW_DIRECTORY.iterdir():

        if not file.is_file():
            continue

        name = file.name.lower()

        if (
            name.startswith("weather")
            and name.endswith(".csv")
        ):
            files.append(file)

    return files

# =============================================================================
# SENSOR : UN FICHIER EST-IL DISPONIBLE ?
# =============================================================================
def raw_csv_available():
    files = get_raw_csv_files()

    logger.info(
        {
            "event": "raw_csv_available",
            "nb_file": len(files),
            "folder": str(RAW_DIRECTORY),
        }
    )
    return len(files) > 0
    
    



# =============================================================================
# SÉLECTION ET VERROUILLAGE D'UN FICHIER
# =============================================================================
def find_raw_file(**context):
    """
    Sélectionne le fichier RAW le plus ancien et le verrouille.

    Exemple :

        weather_batch_001.csv
                    ↓
        weather_batch_001.csv.processing

    Le fichier .processing n'est jamais considéré comme disponible
    par get_raw_csv_files().
    """
    files = sorted(
        get_raw_csv_files(),
        key=lambda f: f.stat().st_mtime
    )

    if not files:
        raise Exception(
            "Aucun fichier RAW météo trouvé"
        )

    for file in files:
        processing_file = Path(
            str(file) + ".processing"
        )

        # Le fichier est déjà en cours de traitement
        if processing_file.exists():
            logger.warning(
                {
                    "event": "file_already_processing",
                    "file": str(file)
                }
            )
            continue

        file.rename(processing_file)

        logger.info(
            {
                "event": "file_locked",
                "source": str(file),
                "processing": str(processing_file)
            }
        )

        return processing_file.name

    raise Exception(
        "Tous les fichiers weather sont déjà en traitement"
    )


# ==========================================================
# Boucle : re-déclenchement du DAG tant que des fichiers restent
# ==========================================================
def more_files_pending(**context):
    """
    Vérifie s'il reste des fichiers RAW non verrouillés dans le dossier.
    Utilisé par un ShortCircuitOperator : si True, la tâche suivante
    (wait_before_retrigger, puis trigger_next_run) s'exécute et redéclenche
    un nouveau run du DAG après un court délai, sans attendre le prochain
    créneau du schedule. Si False, la chaîne s'arrête ici et c'est le
    schedule périodique qui reprendra la surveillance plus tard.
                              

    trigger_rule="all_done" sur cette tâche (voir plus bas) garantit
    qu'on vérifie systématiquement, même si une intégration a échoué,
    pour ne pas laisser les fichiers restants en attente indéfiniment.

    Le délai anti-surcharge n'est PLUS géré ici par un time.sleep()
    bloquant (bug : bloquait un worker Airflow + "time" n'était même pas
    importé). Il est désormais délégué à wait_before_retrigger, un
    TimeDeltaSensor en mode "reschedule" qui ne consomme pas de slot
    worker pendant l'attente.
    """
    pending = raw_csv_available()

    logger.info(
        {
            "event": "more_files_pending_check",
            "pending": pending,
        }
    )

    return pending


# ----------------
# -----------------------------
# DÉFINITION DU DAG
# -----------------------------
with DAG(
    dag_id="weather_csv_integration",
    default_args=default_args,
    description="Intégration continue des données météo",
    start_date=datetime(2026, 6, 1),
    # Un seul run actif à la fois.
    max_active_runs=1,
    # -------------------------------------------------------------------------
    # Surveillance périodique
    #
    # Si aucun fichier n'est disponible, le run se termine.
    # Le prochain run reprendra la surveillance dans 1 minute.
    # -------------------------------------------------------------------------
    schedule=timedelta(minutes=1),
    catchup=False,
    tags=["mlops26", "weather", "data-integration"],
) as dag:

    logger.info(
        {
            "event": "dag_registering",
            "message": "DAG weather_csv_integration",
            "schedule": "1 minute",
            "max_active_runs": 1,
        }
    )

    # -----------------------------------------
    # Etape 1 - Attente détection des csv
    # -----------------------------------------
    wait_for_csv = PythonSensor(
        task_id="wait_for_raw_csv",
        python_callable= raw_csv_available,
        poke_interval=30,
        timeout=180,
        mode="reschedule",
        # Absence de fichier = fin normale de ce run.
        soft_fail=True,
    )

    # -----------------------------------------
    # Etape 2 - Identification et verrouillage de csv
    # Ne pousse en XCom que le NOM du fichier .processing
    # -----------------------------------------
    find_file = PythonOperator(
        task_id="find_raw_file",
        python_callable=find_raw_file,
        do_xcom_push=True
    )


    # -----------------------------------------
    # Etape 3 - Ingestion des donnees
    # Charge le CSV dans la base PostgreSQL
    # -----------------------------------------
    integration_weather_data = DockerOperator(
        task_id="load_weather_csv",
        image=SETTINGS["docker"]["data-integration_image"],
        command=[
            "python",
            "/app/weather_ingestion_entrypoint.py",
            "--file",
            "{{ ti.xcom_pull(task_ids='find_raw_file') }}",
        ],
        on_failure_callback=rollback_processing_file,
        pool="weather_pool",
        environment={
            "POSTGRES_WTH_HOST": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).host }}",
            "POSTGRES_WTH_PORT": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).port }}",
            "POSTGRES_WTH_DB": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).schema }}",
            "POSTGRES_WTH_USER": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).login }}",
            "POSTGRES_WTH_PASSWORD": "{{ conn.get(var.value.WEATHER_POSTGRES_CONN_ID).password }}",
            "AIRFLOW_CONTAINER_DATA_DIR": os.environ["AIRFLOW_CONTAINER_DATA_DIR"],
            "AIRFLOW_CONTAINER_LOGS_DIR": os.environ["AIRFLOW_CONTAINER_LOGS_DIR"],
            "WEATHER_HOST_DATA_DIR": os.environ["WEATHER_HOST_DATA_DIR"],
            "WEATHER_HOST_LOGS_DIR": os.environ["WEATHER_HOST_LOGS_DIR"],
            "WEATHER_HOST_MODELS_DIR":os.environ["WEATHER_HOST_MODELS_DIR"], 
            "WEATHER_HOST_METRICS_DIR":os.environ["WEATHER_HOST_METRICS_DIR"],
        },
        docker_url="unix://var/run/docker.sock",
        network_mode=NETWORK_MODE,
        mounts=mounts,
        # Limites CPU & RAM
        mem_limit="2g",
        cpus=1,
        auto_remove="force",
        mount_tmp_dir=False,
        do_xcom_push=False,
    )


    # -----------------------------------------
    # Etape 4 - Boucle : s'il reste des fichiers RAW non traités
    # (arrivés pendant ce run, ou pas encore verrouillés), on redéclenche
    # le DAG (après le délai anti-surcharge de l'étape 5) au lieu d'attendre
    # le prochain créneau du schedule. trigger_rule="all_done" : on vérifie
    # même si une des tâches d'intégration a échoué, pour ne pas bloquer les
    # autres fichiers en attente.
    # -----------------------------------------
    check_for_more_files = ShortCircuitOperator(
        task_id="check_for_more_files",
        python_callable=more_files_pending,
        trigger_rule="all_done",
    )

    # =========================================================================
    # ÉTAPE 5 - ATTENTE ANTI-SURCHARGE
    # =========================================================================
    #
    # Le délai est volontairement séparé de more_files_pending().
    #
    # Ainsi :
    #
    #     check_for_more_files
    #             ↓
    #     wait_before_retrigger
    #             ↓
    #        trigger_next_run
    #
                                                    
     
    # On évite time.sleep() dans une tâche Python : ce sensor tourne en
    # mode "reschedule", il libère donc le worker pendant l'attente.
                                              
                                                                        
    # =========================================================================

                                                                             

    wait_before_retrigger = TimeDeltaSensor(
        task_id="wait_before_retrigger",

        delta=timedelta(
            seconds=RETRIGGER_POKE_INTERVAL_SECONDS
        ),

        mode="reschedule",
    )

    trigger_next_run = TriggerDagRunOperator(
        task_id="trigger_next_run",
        trigger_dag_id="weather_csv_integration",
        wait_for_completion=False,
    )

    wait_for_csv >> find_file >> integration_weather_data
    integration_weather_data >> check_for_more_files >> wait_before_retrigger >> trigger_next_run


    
