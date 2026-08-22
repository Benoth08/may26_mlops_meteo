#!/usr/bin/env python3
"""
===============================================================================
Projet : mai26_continu_mlops / WeatherAUS
-------------------------------------------------------------------------------

Étape 3 — Promotion du modèle.

Utilise l'alias MLflow "champion" (cf. CHAMPION_ALIAS) plutôt que les
stages du Model Registry (None/Staging/Production) : ces derniers sont
dépréciés depuis MLflow 2.9 et seront supprimés dans une future version
majeure, au profit des alias et tags de version.

Responsabilités :
    - Récupérer la dernière version du modèle enregistrée dans MLflow
    - Récupérer la version actuellement taguée "champion"
    - Comparer la métrique principale
    - Promouvoir le nouveau modèle (réassigner l'alias) uniquement si
      l'amélioration minimale définie dans params.yaml est atteinte
    - Conserver le champion actuel dans le cas contraire
    - Alerter (échec de la tâche) si le F1 du modèle réellement déployé
      tombe sous un seuil absolu (F1_ALERT_THRESHOLD), indépendamment
      de la métrique utilisée pour décider de la promotion

Règle de promotion :
    nouveau_score - score_champion >= min_score_improvement
===============================================================================
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import mlflow
import yaml

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from core.settings import SETTINGS
from core.logger import get_logger
from core.params import load_params
from core.config import load_postgres_config

from training_state import TrainingStateRepository, build_engine


# =============================================================================
# LOGGER
# =============================================================================
logger = get_logger("promote_model")

# =============================================================================
# CONFIGURATION
# =============================================================================
REGISTERED_MODEL_NAME = SETTINGS["models"]["registered_model_name"]
METRICS_FILE = Path(SETTINGS["paths"]["metrics"]) / SETTINGS["models"]["metrics"]

# Alias remplaçant le stage "Production" (déprécié, cf. docstring du module).
CHAMPION_ALIAS = "champion"

# Modèle candidat (sortie de train_model.py/evaluate_model.py) vs modèle
# réellement servi par l'API et predict_model.py. Ce script est le seul
# point qui copie l'un vers l'autre, et ne le fait que si la promotion
# est acceptée : c'est ce qui fait de la décision de promotion un vrai
# gate de déploiement, plutôt qu'une simple étiquette MLflow sans effet.
CANDIDATE_MODEL_PATH = (
    Path(SETTINGS["paths"]["models"]) / SETTINGS["models"]["candidate_model"]
)
SERVING_MODEL_PATH = (
    Path(SETTINGS["paths"]["models"]) / SETTINGS["models"]["model"]
)

# Seuil minimal de F1 attendu pour le modèle réellement déployé.
# Vérifié indépendamment de primary_metric (filet de sécurité en valeur absolue).
F1_ALERT_THRESHOLD = float(os.environ.get("F1_ALERT_THRESHOLD", "0.55"))


# =============================================================================
# PARAMS
# =============================================================================
def get_training_rules(params: dict) -> tuple[str, float]:
    """
    Récupère les règles de promotion.

    Returns:
        primary_metric,
        min_score_improvement
    """

    evaluation = params.get("evaluation", {})
    training = params.get("training", {})
    primary_metric = evaluation.get("primary_metric", "f1")
    min_score_improvement = float(training.get("min_score_improvement", 0.01))

    if min_score_improvement < 0:
        raise ValueError(
            "training.min_score_improvement "
            "doit être >= 0."
        )

    return (
        primary_metric,
        min_score_improvement,
    )


# =============================================================================
# MLflow
# =============================================================================
def configure_mlflow() -> None:
    """
    Configure la connexion MLflow à partir des variables d'environnement.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

        logger.info(
            {
                "event": "mlflow_tracking_configured",
                "tracking_uri": tracking_uri,
            }
        )

    else:
        logger.warning(
            {
                "event": "mlflow_tracking_uri_missing",
                "message": (
                    "MLFLOW_TRACKING_URI "
                    "n'est pas définie."
                ),
            }
        )


# =============================================================================
# VERSIONS MLflow
# =============================================================================
def get_latest_version(client: MlflowClient):
    """
    Récupère la version la plus récente enregistrée pour le modèle.

    evaluate_model.py doit avoir enregistré cette version dans le
    Model Registry (mlflow.sklearn.log_model(..., registered_model_name=...)).

    Les stages étant dépréciés, il n'existe plus de notion de version
    "sans stage" : on prend simplement le numéro de version le plus élevé.
    """

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not versions:
        return None

    return max(versions, key=lambda version: int(version.version))


def get_champion_version(client: MlflowClient):
    """
    Récupère la version actuellement taguée avec l'alias CHAMPION_ALIAS
    (équivalent de l'ancien stage "Production").

    Retourne None si le modèle n'a pas encore de champion (premier
    entraînement, ou alias jamais assigné).
    """

    # DagsHub renvoie INVALID_PARAMETER_VALUE (et non RESOURCE_DOES_NOT_EXIST,
    # comme le ferait un serveur MLflow standard) quand l'alias n'a encore
    # jamais été assigné à ce modèle : les deux codes signifient donc
    # "pas de champion actuel" ici.
    NO_CHAMPION_ERROR_CODES = {
        "RESOURCE_DOES_NOT_EXIST",
        "INVALID_PARAMETER_VALUE",
    }

    try:
        return client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            CHAMPION_ALIAS,
        )

    except MlflowException as error:
        if getattr(error, "error_code", None) in NO_CHAMPION_ERROR_CODES:
            return None
        raise


# =============================================================================
# MÉTRIQUES MLflow
# =============================================================================
def metric_of(
    client: MlflowClient,
    version,
    metric_name: str,
) -> float:
    """
    Récupère une métrique depuis le run MLflow associé
    à une version du modèle.
    """

    run = client.get_run(version.run_id)
    value = run.data.metrics.get(metric_name)

    if value is None:
        raise ValueError(
            f"La métrique '{metric_name}' "
            f"est absente du run MLflow "
            f"{version.run_id}."
        )

    return float(value)


# =============================================================================
# SCORES.JSON
# =============================================================================
def load_local_score(metric_name: str) -> float | None:
    """
    Lit éventuellement metrics/scores.json.

    Ce fichier constitue un contrôle local supplémentaire.

    Le score MLflow reste la source utilisée pour la promotion
    du modèle enregistré.
    """

    if not METRICS_FILE.exists():
        logger.warning(
            {
                "event": "scores_file_missing",
                "path": str(
                    METRICS_FILE
                ),
            }
        )
        return None

    try:
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            scores = json.load(file)

        if ("metrics" in scores and metric_name in scores["metrics"]):
            return float(scores["metrics"][metric_name])

        if ("primary_score" in scores and scores.get("primary_metric") == metric_name):
            return float(scores["primary_score"])

        if metric_name in scores:
            return float(scores[metric_name])

    except Exception as error:
        logger.warning(
            {
                "event": "scores_file_read_failed",
                "error": str(error),
            }
        )

    return None


# =============================================================================
# ALERTE F1 (filet de sécurité)
# =============================================================================
def alert_if_below_f1_threshold(
    client: MlflowClient,
    version,
    context: str,
) -> None:
    """
    Alerte (et fait échouer la tâche Airflow) si le F1 du modèle réellement
    déployé tombe sous F1_ALERT_THRESHOLD, indépendamment de la métrique
    utilisée pour décider de la promotion (primary_metric).
    """

    try:
        f1 = metric_of(client=client, version=version, metric_name="f1")
    except Exception as error:
        logger.warning(
            {
                "event": "f1_alert_check_failed",
                "version": str(version.version),
                "error": str(error),
            }
        )
        return

    if f1 < F1_ALERT_THRESHOLD:
        logger.error(
            {
                "event": "f1_sous_seuil",
                "f1": f1,
                "seuil": F1_ALERT_THRESHOLD,
                "contexte": context,
            }
        )
        print(
            "ALERTE : F1 de {:.4f} sous le seuil minimal de {:.4f} ({})".format(
                f1, F1_ALERT_THRESHOLD, context
            )
        )
        sys.exit(1)


# =============================================================================
# COMPARAISON
# =============================================================================
def compare_scores(
    new_score: float,
    champion_score: float | None,
    min_score_improvement: float,
) -> tuple[bool, float | None, str]:
    """
    Applique la règle de promotion.

    Returns:
        promote,
        improvement,
        reason
    """

    # -------------------------------------------------------------------------
    # Aucun champion actuel
    # -------------------------------------------------------------------------
    if champion_score is None:
        return (True, None, "first_champion_model")

    improvement = new_score - champion_score

    # -------------------------------------------------------------------------
    # Règle de promotion
    # -------------------------------------------------------------------------
    promote = improvement >= min_score_improvement
    if promote:
        reason = "minimum_improvement_reached"

    else:
        reason = "improvement_not_sufficient"

    return (promote, improvement, reason)


# =============================================================================
# PROMOTION
# =============================================================================
def deploy_candidate_to_serving() -> None:
    """
    Copie le modèle candidat (CANDIDATE_MODEL_PATH) vers le fichier
    réellement chargé par le serving (SERVING_MODEL_PATH) : predict_model.py
    et l'API le lisent directement, sans passer par MLflow.

    À appeler uniquement après une promotion acceptée : c'est ce qui
    relie la décision de promotion à ce qui est effectivement servi.
    """

    if not CANDIDATE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle candidat introuvable : {CANDIDATE_MODEL_PATH}"
        )

    SERVING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(CANDIDATE_MODEL_PATH, SERVING_MODEL_PATH)

    logger.info(
        {
            "event": "candidate_deployed_to_serving",
            "source": str(CANDIDATE_MODEL_PATH),
            "destination": str(SERVING_MODEL_PATH),
        }
    )


def promote_version(
    client: MlflowClient,
    version,
) -> None:
    """
    Réassigne l'alias CHAMPION_ALIAS à cette version.

    Un alias est unique par modèle enregistré : le réassigner retire
    automatiquement l'alias de l'ancienne version championne, sans
    action supplémentaire.
    """

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=version.version,
    )


# =============================================================================
# TAGS MLflow
# =============================================================================

def set_promotion_tags(
    client: MlflowClient,
    version,
    primary_metric: str,
    new_score: float,
    champion_score: float | None,
    improvement: float | None,
    min_score_improvement: float,
    reason: str,
) -> None:
    """
    Ajoute les informations de promotion au modèle.
    """

    tags = {
        "promotion_status": CHAMPION_ALIAS,
        "promotion_metric": primary_metric,
        "promotion_score": str(
            new_score
        ),
        "promotion_min_improvement": str(
            min_score_improvement
        ),
        "promotion_reason": reason,
    }

    if champion_score is not None:
        tags["champion_previous_score"] = str(champion_score)

    if improvement is not None:
        tags["promotion_improvement"] = str(improvement)

    for key, value in tags.items():
        try:
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=version.version,
                key=key,
                value=value,
            )

        except Exception as error:
            logger.warning(
                {
                    "event": "mlflow_tag_failed",
                    "key": key,
                    "error": str(error),
                }
            )


# =============================================================================
# ÉTAT D'ENTRAÎNEMENT (model_training_status)
# =============================================================================

def record_training_attempt(
    params: dict,
    promoted: bool,
    new_version,
    new_score: float,
    champion,
    champion_score: float | None,
) -> None:
    """
    Enregistre cette tentative d'entraînement dans model_training_status.

    C'est le seul point qui appelle TrainingStateRepository.update_training_status() :
    sans lui, weather_training_trigger relit indéfiniment un état jamais
    mis à jour et re-déclenche un entraînement à chaque créneau horaire,
    quel que soit le résultat des tentatives précédentes.

    last_training_date / last_training_row_count avancent à chaque
    tentative, promue ou non : c'est ce qui empêche weather_training_trigger
    de re-déclencher en boucle sur les mêmes données. last_model_version /
    last_score reflètent en revanche le modèle réellement déployé (le
    nouveau si promu, le champion actuel sinon).
    """

    model_name = (
        params.get("mlflow", {}).get("model_name", "rain_prediction")
    )
    training_row_count = int(os.getenv("TRAINING_ROW_COUNT", "0"))
    training_reason = os.getenv("TRAINING_REASON", "manual")

    if promoted:
        deployed_version = new_version.version
        deployed_score = new_score
    elif champion is not None:
        deployed_version = champion.version
        deployed_score = champion_score
    else:
        deployed_version = None
        deployed_score = None

    cfg = load_postgres_config()
    engine = build_engine(cfg.sqlalchemy_uri)
    repository = TrainingStateRepository(engine=engine)

    repository.update_training_status(
        model_name=model_name,
        training_row_count=training_row_count,
        model_version=deployed_version,
        score=deployed_score,
        status="promoted" if promoted else "rejected",
        training_reason=training_reason,
    )

    logger.info(
        {
            "event": "training_status_updated",
            "model_name": model_name,
            "training_row_count": training_row_count,
            "training_reason": training_reason,
            "deployed_version": deployed_version,
            "deployed_score": deployed_score,
            "status": "promoted" if promoted else "rejected",
        }
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    logger.info(
        {
            "event": "promotion_started",
            "model_name": (
                REGISTERED_MODEL_NAME
            ),
        }
    )

    # =========================================================================
    # 1. Paramètres
    # =========================================================================
    try:
        params = load_params()
        primary_metric, min_score_improvement = get_training_rules(params)

    except Exception as error:
        logger.error(
            {
                "event": "params_loading_failed",
                "error": str(error),
            },
            exc_info=True,
        )
        sys.exit(1)

    logger.info(
        {
            "event": "promotion_configuration",
            "primary_metric": primary_metric,
            "min_score_improvement": min_score_improvement
        }
    )

    # =========================================================================
    # 2. MLflow
    # =========================================================================
    try:
        configure_mlflow()
        client = MlflowClient()

    except Exception as error:
        logger.error(
            {
                "event": "mlflow_client_failed",
                "error": str(error),
            },
            exc_info=True,
        )
        sys.exit(1)

    # =========================================================================
    # 3. Nouveau modèle
    # =========================================================================
    try:
        new = get_latest_version(client)

    except Exception as error:
        logger.error(
            {
                "event": "new_version_lookup_failed",
                "error": str(error),
            },
            exc_info=True,
        )
        sys.exit(1)

    if new is None:
        logger.info(
            {
                "event": "no_new_model",
                "message": (
                    "Aucune version enregistrée "
                    "dans le Model Registry."
                ),
            }
        )

        print("ℹ️ Aucune nouvelle version à promouvoir.")

        return

    # =========================================================================
    # 4. Modèle champion actuel
    # =========================================================================
    try:
        champion = get_champion_version(client)

    except Exception as error:
        logger.error(
            {
                "event": "champion_version_lookup_failed",
                "error": str(error),
            },
            exc_info=True,
        )
        sys.exit(1)

    if champion is not None and champion.version == new.version:
        logger.info(
            {
                "event": "no_new_model",
                "message": (
                    "La dernière version enregistrée est déjà "
                    f"l'alias {CHAMPION_ALIAS}."
                ),
                "version": str(new.version),
            }
        )

        print("ℹ️ Aucune nouvelle version à promouvoir.")

        return

    # =========================================================================
    # 5. Score nouveau modèle
    # =========================================================================
    try:
        new_score = metric_of(
            client=client,
            version=new,
            metric_name=primary_metric,
        )

    except Exception as error:
        logger.error(
            {
                "event": "new_model_metric_missing",
                "version": str(new.version),
                "metric": primary_metric,
                "error": str(error),
            },
            exc_info=True,
        )
        sys.exit(1)

    # =========================================================================
    # 6. Contrôle scores.json
    # =========================================================================
    local_score = load_local_score(primary_metric)
    if local_score is not None:
        logger.info(
            {
                "event": "local_score_found",
                "metric": primary_metric,
                "mlflow_score": new_score,
                "local_score": local_score
            }
        )

        # ---------------------------------------------------------------------
        # On vérifie que le fichier produit par evaluate_model correspond
        # au modèle que nous sommes en train de promouvoir.
        # ---------------------------------------------------------------------
        if abs(new_score - local_score) > 1e-6:
            logger.error(
                {
                    "event": "score_mismatch",
                    "mlflow_score": new_score,
                    "local_score": local_score,
                    "message": "Le score MLflow et scores.json sont différents."
                }
            )
            sys.exit(1)

    # =========================================================================
    # 7. Score du champion actuel
    # =========================================================================
    champion_score = None
    if champion is not None:
        try:
            champion_score = metric_of(
                client=client,
                version=champion,
                metric_name=primary_metric,
            )

        except Exception as error:
            logger.error(
                {
                    "event": "champion_metric_missing",
                    "version": str(champion.version),
                    "metric": primary_metric,
                    "error": str(error)
                },
                exc_info=True,
            )
            sys.exit(1)

    # =========================================================================
    # 8. Comparaison
    # =========================================================================
    promote, improvement, reason = compare_scores(
        new_score=new_score,
        champion_score=champion_score,
        min_score_improvement=min_score_improvement)

    logger.info(
        {
            "event": "model_comparison",
            "new_version": str(new.version),
            "champion_version": (
                str(
                    champion.version
                )
                if champion
                else None
            ),
            "metric": primary_metric,
            "new_score": new_score,
            "champion_score": champion_score,
            "improvement": improvement,
            "min_score_improvement": min_score_improvement,
            "promote": promote,
            "reason": reason,
        }
    )

    # =========================================================================
    # 9. Promotion
    # =========================================================================
    if promote:
        try:
            promote_version(client=client, version=new)

            deploy_candidate_to_serving()

            set_promotion_tags(
                client=client,
                version=new,
                primary_metric=(
                    primary_metric
                ),
                new_score=new_score,
                champion_score=(
                    champion_score
                ),
                improvement=improvement,
                min_score_improvement=(
                    min_score_improvement
                ),
                reason=reason,
            )

        except Exception as error:
            logger.error(
                {
                    "event": "promotion_failed",
                    "version": str(new.version),
                    "error": str(error),
                },
                exc_info=True,
            )
            sys.exit(1)

        # ---------------------------------------------------------------------
        # Succès
        # ---------------------------------------------------------------------
        print("========================================")
        print(f"✅ MODÈLE PROMU (alias {CHAMPION_ALIAS})")
        print("========================================")
        print(f"Version       : {new.version}")
        print(f"Métrique      : {primary_metric}")
        print(f"Nouveau score : {new_score:.4f}")

        if champion_score is not None:
            print(f"Ancien score  : {champion_score:.4f}")
            print(f"Amélioration  : {improvement:.4f}")

        print(f"Seuil minimum : {min_score_improvement:.4f}")
        print(f"Raison        : {reason}")
        print("========================================")

    else:
        # ---------------------------------------------------------------------
        # Rejet
        # ---------------------------------------------------------------------
        print("========================================")
        print("ℹ️ MODÈLE NON PROMU")
        print("========================================")
        print(f"Version       : {new.version}")
        print(f"Métrique      : {primary_metric}")
        print(f"Nouveau score : {new_score:.4f}")
        print(f"Score champion: {champion_score:.4f}")
        print(f"Amélioration  : {improvement:.4f}")
        print(f"Seuil minimum : {min_score_improvement:.4f}")
        print(f"Raison        : {reason}")
        print("Le modèle champion actuel reste inchangé")
        print("========================================")

    # =========================================================================
    # 9bis. Alerte F1 (filet de sécurité, indépendant de primary_metric)
    # =========================================================================
    deployed_for_alert = new if promote else champion
    if deployed_for_alert is not None:
        alert_if_below_f1_threshold(
            client=client,
            version=deployed_for_alert,
            context="nouveau modele promu" if promote else "modele champion conserve",
        )

    # =========================================================================
    # 10. Mise à jour de l'état d'entraînement (model_training_status)
    # =========================================================================
    try:
        record_training_attempt(
            params=params,
            promoted=promote,
            new_version=new,
            new_score=new_score,
            champion=champion,
            champion_score=champion_score,
        )

    except Exception as error:
        # Un échec d'enregistrement de l'état d'entraînement ne doit pas
        # remettre en cause une promotion déjà effectuée : on logue et on
        # laisse le run se terminer normalement.
        logger.error(
            {
                "event": "training_status_update_failed",
                "error": str(error),
            },
            exc_info=True,
        )


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    main()
