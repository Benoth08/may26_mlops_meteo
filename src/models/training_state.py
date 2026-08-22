"""
Gestion de l'état des entraînements du modèle.

Ce module centralise la lecture et la mise à jour de la table
model_training_status.

Responsabilité :
    - récupérer le nombre de lignes du dataset ;
    - récupérer le dernier entraînement ;
    - enregistrer le résultat d'un entraînement.

La logique de décision (doit-on entraîner ?) est dans
training_condition.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class TrainingStatus:
    """État du dernier entraînement d'un modèle."""

    model_name: str
    last_training_date: Optional[datetime]
    last_training_row_count: int
    last_model_version: Optional[str]
    last_score: Optional[float]
    status: Optional[str]
    training_reason: Optional[str]


class TrainingStateRepository:
    """
    Repository d'accès à la table model_training_status.

    Aucun appel Airflow ne doit être effectué dans cette classe.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_current_row_count(
        self,
        table_name: str = "weather_data_raw",
    ) -> int:
        """
        Retourne le nombre actuel de lignes du dataset.

        Le nom de table doit rester contrôlé par la configuration
        interne du projet et ne doit pas provenir directement
        d'une entrée utilisateur.
        """

        query = text(
            f"""
            SELECT COUNT(*)
            FROM {table_name}
            """
        )

        with self.engine.connect() as connection:
            result = connection.execute(query)
            return int(result.scalar_one())

    def get_last_training(
        self,
        model_name: str,
    ) -> Optional[TrainingStatus]:
        """Retourne le dernier état connu du modèle."""

        query = text(
            """
            SELECT
                model_name,
                last_training_date,
                last_training_row_count,
                last_model_version,
                last_score,
                status,
                training_reason
            FROM model_training_status
            WHERE model_name = :model_name
            """
        )

        with self.engine.connect() as connection:
            row = connection.execute(
                query,
                {"model_name": model_name},
            ).mappings().first()

        if row is None:
            return None

        return TrainingStatus(
            model_name=row["model_name"],
            last_training_date=row["last_training_date"],
            last_training_row_count=int(
                row["last_training_row_count"] or 0
            ),
            last_model_version=row["last_model_version"],
            last_score=(
                float(row["last_score"])
                if row["last_score"] is not None
                else None
            ),
            status=row["status"],
            training_reason=row["training_reason"],
        )

    def update_training_status(
        self,
        model_name: str,
        training_row_count: int,
        model_version: Optional[str],
        score: Optional[float],
        status: str,
        training_reason: Optional[str] = None,
    ) -> None:
        """
        Met à jour l'état du modèle.

        Une seule ligne est conservée par model_name.
        """

        now = datetime.now(timezone.utc)

        query = text(
            """
            INSERT INTO model_training_status (
                model_name,
                last_training_date,
                last_training_row_count,
                last_model_version,
                last_score,
                status,
                training_reason,
                updated_at
            )
            VALUES (
                :model_name,
                :last_training_date,
                :last_training_row_count,
                :last_model_version,
                :last_score,
                :status,
                :training_reason,
                :updated_at
            )
            ON CONFLICT (model_name)
            DO UPDATE SET
                last_training_date = EXCLUDED.last_training_date,
                last_training_row_count = EXCLUDED.last_training_row_count,
                last_model_version = EXCLUDED.last_model_version,
                last_score = EXCLUDED.last_score,
                status = EXCLUDED.status,
                training_reason = EXCLUDED.training_reason,
                updated_at = EXCLUDED.updated_at
            """
        )

        with self.engine.begin() as connection:
            connection.execute(
                query,
                {
                    "model_name": model_name,
                    "last_training_date": now,
                    "last_training_row_count": training_row_count,
                    "last_model_version": model_version,
                    "last_score": score,
                    "status": status,
                    "training_reason": training_reason,
                    "updated_at": now,
                },
            )


def build_engine(database_url: str) -> Engine:
    """
    Crée l'engine SQLAlchemy.

    database_url doit être fourni par la configuration
    de l'application et non codé en dur.
    """

    return create_engine(
        database_url,
        pool_pre_ping=True,
        future=True,
    )