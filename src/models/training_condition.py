"""
Détermination de la nécessité de lancer un nouvel entraînement.

Règles :

    0. Si la base ne contient aucune donnée (current_row_count == 0) :
           pas d'entraînement, quel que soit l'historique.

    1. Si suffisamment de nouvelles lignes sont disponibles :
           new_rows >= min_new_rows

    OU

    2. Si le délai maximal depuis le dernier entraînement est dépassé :
           elapsed_hours >= max_interval_hours

Alors un entraînement est déclenché.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class TrainingDecision:
    """Résultat de la décision de déclenchement."""

    should_train: bool
    reason: str
    current_row_count: int
    last_training_row_count: int
    new_rows: int
    elapsed_hours: Optional[float]


def evaluate_training_condition(
    current_row_count: int,
    last_training_row_count: int,
    last_training_date: Optional[datetime],
    min_new_rows: int,
    max_interval_hours: int,
    now: Optional[datetime] = None,
) -> TrainingDecision:
    """
    Détermine si un entraînement doit être lancé.

    Conditions :

        new_rows >= min_new_rows
        OU
        elapsed_hours >= max_interval_hours

    Si aucun entraînement n'a encore été effectué, le training
    est déclenché immédiatement.
    """

    if current_row_count < 0:
        raise ValueError(
            "current_row_count ne peut pas être négatif."
        )

    if last_training_row_count < 0:
        raise ValueError(
            "last_training_row_count ne peut pas être négatif."
        )

    if min_new_rows <= 0:
        raise ValueError(
            "min_new_rows doit être strictement positif."
        )

    if max_interval_hours <= 0:
        raise ValueError(
            "max_interval_hours doit être strictement positif."
        )

    new_rows = max(
        0,
        current_row_count - last_training_row_count,
    )

    # Base vide : aucun dataset disponible, pas d'entraînement possible,
    # même s'il s'agirait d'un premier entraînement. Évite un déclenchement
    # au tout premier passage du trigger, avant toute intégration de données.
    if current_row_count == 0:
        return TrainingDecision(
            should_train=False,
            reason="no_data",
            current_row_count=current_row_count,
            last_training_row_count=last_training_row_count,
            new_rows=new_rows,
            elapsed_hours=None,
        )

    # Premier entraînement
    if last_training_date is None:
        return TrainingDecision(
            should_train=True,
            reason="first_training",
            current_row_count=current_row_count,
            last_training_row_count=last_training_row_count,
            new_rows=new_rows,
            elapsed_hours=None,
        )

    if now is None:
        now = datetime.now(timezone.utc)

    # Gestion des dates naïves pour éviter les erreurs de comparaison
    if last_training_date.tzinfo is None:
        last_training_date = last_training_date.replace(
            tzinfo=timezone.utc
        )

    elapsed_seconds = (
        now - last_training_date
    ).total_seconds()

    elapsed_hours = elapsed_seconds / 3600

    # Priorité à la condition de volume
    if new_rows >= min_new_rows:
        return TrainingDecision(
            should_train=True,
            reason="new_rows_threshold",
            current_row_count=current_row_count,
            last_training_row_count=last_training_row_count,
            new_rows=new_rows,
            elapsed_hours=elapsed_hours,
        )

    # Puis condition temporelle
    if elapsed_hours >= max_interval_hours:
        return TrainingDecision(
            should_train=True,
            reason="max_interval",
            current_row_count=current_row_count,
            last_training_row_count=last_training_row_count,
            new_rows=new_rows,
            elapsed_hours=elapsed_hours,
        )

    return TrainingDecision(
        should_train=False,
        reason="threshold_not_reached",
        current_row_count=current_row_count,
        last_training_row_count=last_training_row_count,
        new_rows=new_rows,
        elapsed_hours=elapsed_hours,
    )