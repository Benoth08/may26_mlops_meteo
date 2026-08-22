#!/usr/bin/env python3
"""
===============================================================================
    mai26_continu_mlops / projet Weather
    ---------------------------------------------------------------------------
    DAG : weather_training_trigger

    Responsabilité :
        Déterminer si un nouvel entraînement est nécessaire puis déclencher
        le DAG weather_models.

    Stratégie :
        NO TRAIN si :
            - la base ne contient aucune donnée (current_row_count == 0)

        Sinon, TRAIN si :
            - premier entraînement
            OU
            - nouvelles lignes >= training.min_new_rows
            OU
            - temps depuis le dernier entraînement >= training.max_interval_hours

    Version :
        1.0.0
===============================================================================
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.standard.operators.trigger_dagrun import (
    TriggerDagRunOperator,
)
from airflow.sdk import Variable

from models.training_condition import evaluate_training_condition
from models.training_state import TrainingStateRepository

from core.settings import SETTINGS
from core.params import load_params

# =============================================================================
# LOGGER
# =============================================================================

logger = logging.getLogger("airflow.task.weather_training_trigger")


# =============================================================================
# CONFIGURATION
# =============================================================================


PARAMS = load_params()

POSTGRES_CONNECTION_ID = Variable.get(
    "WEATHER_POSTGRES_CONN_ID",
    default="weather_postgres",
)

DATA_TABLE = SETTINGS["postgres"]["table_raw"]


# =============================================================================
# CALLBACK ERREUR
# =============================================================================
def log_training_trigger_failure(context):
    """
    Callback exécuté lorsqu'une tâche du DAG trigger échoue.
    """

    dag_id = context.get("dag").dag_id
    task_id = context.get("task_instance").task_id
    run_id = context.get("run_id")
    exception = str(context.get("exception"))

    logger.error(
        {
            "event": "training_trigger_failed",
            "message": "Échec du DAG de déclenchement du training",
            "dag_id": dag_id,
            "task_id": task_id,
            "run_id": run_id,
            "exception": exception,
        }
    )


# =============================================================================
# CONFIGURATION PAR DÉFAUT
# =============================================================================
default_args = {
    "owner": "weather",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": log_training_trigger_failure,
}


# =============================================================================
# DAG
# =============================================================================

with DAG(
    dag_id="weather_training_trigger",
    default_args=default_args,
    description=(
        "Détermine si un nouvel entraînement est nécessaire "
        "et déclenche weather_models"
    ),

    start_date=datetime(2026, 6, 1),

    # Vérification toutes les heures.
    # Cela ne signifie PAS un entraînement toutes les heures.
    schedule="0 * * * *",

    catchup=False,

    # Un seul trigger à la fois.
    max_active_runs=1,

    tags=[
        "mlops26",
        "weather",
        "training",
        "trigger",
    ],
) as dag:

    # =========================================================================
    # 1. Vérification de la condition d'entraînement
    # =========================================================================

    @task
    def check_training_condition() -> dict:
        """
        Détermine si weather_models doit être déclenché.
        """

        training_params = PARAMS.get(
            "training",
            {},
        )

        min_new_rows = int(
            training_params.get(
                "min_new_rows",
                1000,
            )
        )

        max_interval_hours = int(
            training_params.get(
                "max_interval_hours",
                24,
            )
        )

        model_name = (
            PARAMS
            .get("mlflow", {})
            .get("model_name", "rain_prediction")
        )

        logger.info(
            {
                "event": "training_condition_check",
                "model_name": model_name,
                "min_new_rows": min_new_rows,
                "max_interval_hours": max_interval_hours,
            }
        )

        # ---------------------------------------------------------------------
        # PostgreSQL
        # ---------------------------------------------------------------------

        postgres_hook = PostgresHook(
            postgres_conn_id=POSTGRES_CONNECTION_ID
        )

        engine = postgres_hook.get_sqlalchemy_engine()

        repository = TrainingStateRepository(
            engine=engine
        )

        # ---------------------------------------------------------------------
        # Nombre actuel de lignes
        # ---------------------------------------------------------------------

        current_row_count = (
            repository.get_current_row_count(
                table_name=DATA_TABLE
            )
        )

        # ---------------------------------------------------------------------
        # Dernier training
        # ---------------------------------------------------------------------

        last_training = (
            repository.get_last_training(
                model_name=model_name
            )
        )

        if last_training is None:
            last_training_date = None
            last_training_row_count = 0
        else:
            last_training_date = (
                last_training.last_training_date
            )
            last_training_row_count = (
                last_training.last_training_row_count
            )

        # ---------------------------------------------------------------------
        # Evaluation de la règle
        # ---------------------------------------------------------------------
        decision = evaluate_training_condition(
            current_row_count=current_row_count,
            last_training_row_count=last_training_row_count,
            last_training_date=last_training_date,
            min_new_rows=min_new_rows,
            max_interval_hours=max_interval_hours,
        )

        logger.info(
            {
                "event": "training_decision",
                "should_train": decision.should_train,
                "reason": decision.reason,
                "current_row_count": (
                    decision.current_row_count
                ),
                "last_training_row_count": (
                    decision.last_training_row_count
                ),
                "new_rows": decision.new_rows,
                "elapsed_hours": decision.elapsed_hours,
            }
        )

        return {
            "should_train": decision.should_train,
            "reason": decision.reason,
            "model_name": model_name,
            "current_row_count": decision.current_row_count,
            "last_training_row_count": (
                decision.last_training_row_count
            ),
            "new_rows": decision.new_rows,
            "elapsed_hours": decision.elapsed_hours,
        }

    # =========================================================================
    # 2. Branch
    # =========================================================================

    @task.branch
    def decide_training(decision: dict) -> str:
        """
        Décide si le DAG weather_models doit être déclenché.
        """

        if decision["should_train"]:

            logger.info(
                "Training nécessaire : %s",
                decision["reason"],
            )

            return "trigger_weather_models"

        logger.info(
            "Training non nécessaire : %s",
            decision["reason"],
        )

        return "training_not_required"

    # =========================================================================
    # 3. Trigger du DAG weather_models
    # =========================================================================

    trigger_weather_models = TriggerDagRunOperator(
        task_id="trigger_weather_models",
        trigger_dag_id="weather_models",

        # Transmission du contexte au DAG weather_models.
        conf={
            "training_reason": (
                "{{ ti.xcom_pull("
                "task_ids='check_training_condition'"
                ")['reason'] }}"
            ),

            "new_rows": (
                "{{ ti.xcom_pull("
                "task_ids='check_training_condition'"
                ")['new_rows'] }}"
            ),

            "current_row_count": (
                "{{ ti.xcom_pull("
                "task_ids='check_training_condition'"
                ")['current_row_count'] }}"
            ),

            "model_name": (
                "{{ ti.xcom_pull("
                "task_ids='check_training_condition'"
                ")['model_name'] }}"
            ),
        },
        wait_for_completion=True,
        skip_when_already_exists=True,
        reset_dag_run=False,
    )

    # =========================================================================
    # 4. Pas de training
    # =========================================================================

    @task
    def training_not_required(
        decision: dict,
    ) -> None:
        """
        Fin normale lorsqu'un training n'est pas nécessaire.
        """

        logger.info(
            {
                "event": "training_not_required",
                "reason": decision["reason"],
                "new_rows": decision["new_rows"],
                "elapsed_hours": decision["elapsed_hours"],
            }
        )

    # =========================================================================
    # ORCHESTRATION
    # =========================================================================

    decision = check_training_condition()

    branch = decide_training(
        decision
    )

    no_training = training_not_required(
        decision
    )

    branch >> trigger_weather_models

    branch >> no_training