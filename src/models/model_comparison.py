"""
Comparaison entre un nouveau modèle et le modèle actuellement
utilisé.

La promotion est autorisée uniquement si le nouveau modèle
améliore suffisamment la métrique principale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelComparison:
    """Résultat de la comparaison de deux modèles."""

    new_score: float
    current_score: float | None
    min_score_improvement: float
    improvement: float | None
    promote: bool
    reason: str


def compare_models(
    new_score: float,
    current_score: float | None,
    min_score_improvement: float,
) -> ModelComparison:
    """
    Compare le nouveau modèle au modèle actuellement utilisé.

    Règle :

        new_score >= current_score + min_score_improvement

    Si aucun modèle actuel n'existe, le nouveau modèle peut être
    considéré comme le premier modèle à promouvoir.
    """

    if not 0 <= new_score <= 1:
        raise ValueError(
            "new_score doit être compris entre 0 et 1."
        )

    if current_score is not None and not 0 <= current_score <= 1:
        raise ValueError(
            "current_score doit être compris entre 0 et 1."
        )

    if min_score_improvement < 0:
        raise ValueError(
            "min_score_improvement ne peut pas être négatif."
        )

    # Premier modèle
    if current_score is None:
        return ModelComparison(
            new_score=new_score,
            current_score=None,
            min_score_improvement=min_score_improvement,
            improvement=None,
            promote=True,
            reason="first_model",
        )

    improvement = new_score - current_score

    promote = improvement >= min_score_improvement

    return ModelComparison(
        new_score=new_score,
        current_score=current_score,
        min_score_improvement=min_score_improvement,
        improvement=improvement,
        promote=promote,
        reason=(
            "minimum_improvement_reached"
            if promote
            else "improvement_not_sufficient"
        ),
    )