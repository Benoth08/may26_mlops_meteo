#!/usr/bin/env python3
"""
===============================================================================
    Étape 4 — Évaluation du modèle
    ---------------------------------------------------------------------------

    Évalue le modèle entraîné sur le jeu de test mis de côté.

    Entrées :
        - dataset prétraité contenant X_test / y_test
        - modèle entraîné
        - paramètres optimaux (optionnel)

    Sorties :
        - métriques JSON
        - prédictions CSV
        - run MLflow optionnel
===============================================================================
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import joblib
import pandas as pd

from pathlib import Path

from core.settings import SETTINGS
from core.logger import get_logger
from core.params import load_params

# =============================================================================
# Imports sklearn
# =============================================================================
for _env_key, _env_val in SETTINGS["threads"].items():
    os.environ.setdefault(_env_key, str(_env_val))

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
    confusion_matrix
)


logger = get_logger("evaluate_model")

# =============================================================================
# Chemins
# =============================================================================
MODELS_DIR = Path(SETTINGS["paths"]["models"])
METRICS_DIR = Path(SETTINGS["paths"]["metrics"])
PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
DATA_DIR = Path(SETTINGS["paths"]["data"])

DATASET_PATH = Path(os.getenv("WEATHER_DATASET_PATH", str(PROCESSED_DIR / SETTINGS["models"]["dataset"])))

# Snapshot brut exporté de Postgres (export_dataset.py), journalisé dans MLflow
# avec le modèle pour savoir sur quelles données il a été entraîné.
RAW_DATASET_PATH = Path(os.getenv(
    "WEATHER_RAW_DATASET_PATH",
    str(DATA_DIR / SETTINGS["models"]["raw_dataset"]),
))

# Modèle candidat produit par train_model.py, pas encore promu.
MODEL_PATH = (MODELS_DIR / SETTINGS["models"]["candidate_model"])
METRICS_PATH = (METRICS_DIR / SETTINGS["models"]["candidate_metrics"])
CANDIDATE_INFO_PATH = (MODELS_DIR / SETTINGS["models"]["candidate_info"])
PREDICTIONS_PATH = (DATA_DIR.parent / SETTINGS["models"]["predictions"])
BEST_PARAMS_PATH = (MODELS_DIR / SETTINGS["models"]["best_params"])

REGISTERED_MODEL_NAME = SETTINGS["models"]["registered_model_name"]

# Code de sortie signalant à Airflow (DockerOperator.skip_on_exit_code) qu'il
# n'y a aucune donnée à traiter : la tâche et les tâches en aval doivent être
# marquées "skipped", pas "failed".
SKIP_EXIT_CODE = 99

# =============================================================================
# Paramètres d'évaluation (params.yml, section "evaluation" / "mlflow")
# =============================================================================
PARAMS = load_params()

THRESHOLD = float(PARAMS["evaluation"]["threshold"])
MLFLOW_EXPERIMENT_NAME = SETTINGS["models"]["mlflow_experiment_name"]


# =============================================================================
# Fonctions utilitaires
# =============================================================================
def load_artifacts():
    """
    Charge le dataset de test et le modèle entraîné.

    Returns
    -------
    tuple
        data, model, metadata, best_params
    """

    logger.info(
        {
            "event": "loading_evaluation_artifacts",
            "dataset_path": str(DATASET_PATH),
            "model_path": str(MODEL_PATH),
        }
    )

    try:
        data = joblib.load(DATASET_PATH)

    except FileNotFoundError as exc:
        logger.error(
            {
                "event": "dataset_not_found",
                "path": str(DATASET_PATH),
                "error": str(exc),
            },
            exc_info=True,
        )
        raise

    try:
        artifact = joblib.load(MODEL_PATH)

    except FileNotFoundError as exc:
        logger.error(
            {
                "event": "model_not_found",
                "path": str(MODEL_PATH),
                "error": str(exc),
            },
            exc_info=True,
        )
        raise

    # -------------------------------------------------------------------------
    # Le modèle peut être sauvegardé directement ou dans un dictionnaire.
    # -------------------------------------------------------------------------
    if isinstance(artifact, dict):
        if "pipeline" not in artifact:
            raise KeyError(
                "Le fichier modèle est un dictionnaire mais "
                "ne contient pas la clé 'pipeline'."
            )

        model = artifact["pipeline"]
        metadata = artifact.get("metadata", {})

    else:
        model = artifact
        metadata = {}

    # -------------------------------------------------------------------------
    # best_params est optionnel
    # -------------------------------------------------------------------------
    if BEST_PARAMS_PATH.exists():
        best_params = joblib.load(BEST_PARAMS_PATH)
        if not isinstance(best_params, dict):
            logger.warning(
                {
                    "event": "invalid_best_params_format",
                    "path": str(BEST_PARAMS_PATH),
                }
            )
            best_params = {}

    else:
        logger.warning(
            {
                "event": "best_params_not_found",
                "path": str(BEST_PARAMS_PATH),
                "message": "Evaluation poursuivie sans best_params.",
            }
        )

        best_params = {}

    return data, model, metadata, best_params


def validate_dataset(data):
    """
    Vérifie que le dataset contient les jeux de test nécessaires.
    """
    required_keys = {"X_test", "y_test"}
    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        raise KeyError(
            "Dataset incomplet. Clés manquantes : "
            + ", ".join(sorted(missing_keys))
        )

    X_test = data["X_test"]
    y_test = data["y_test"]

    logger.info(
        {
            "event": "test_dataset_loaded",
            "test_rows": len(X_test),
            "test_features": getattr(X_test, "shape", None),
        }
    )

    return X_test, y_test


def predict(model, X_test):
    """
    Génère les probabilités et les prédictions.

    Le seuil de classification est configurable.
    """

    # -------------------------------------------------------------------------
    # Vérification de predict_proba
    # -------------------------------------------------------------------------
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Le modèle ne fournit pas la méthode predict_proba(). "
            "Impossible de calculer ROC-AUC et PR-AUC."
        )

    # -------------------------------------------------------------------------
    # Probabilités
    # -------------------------------------------------------------------------
    y_proba_all = model.predict_proba(X_test)

    if y_proba_all.ndim != 2:
        raise ValueError(
            "predict_proba() doit retourner un tableau 2D."
        )

    if y_proba_all.shape[1] != 2:
        raise ValueError(
            "Le modèle doit être binaire et retourner deux probabilités."
        )

    # -------------------------------------------------------------------------
    # Identification de la classe positive
    # -------------------------------------------------------------------------
    if not hasattr(model, "classes_"):
        logger.warning(
            {
                "event": "classes_attribute_missing",
                "message": (
                    "Impossible de vérifier explicitement la classe positive. "
                    "La deuxième colonne de predict_proba() sera utilisée."
                ),
            }
        )

        positive_column = 1

    else:
        classes = list(model.classes_)

        logger.info(
            {
                "event": "model_classes",
                "classes": [str(value) for value in classes],
            }
        )

        # Pour un modèle binaire, la deuxième classe est normalement la classe
        # positive. Cela fonctionne notamment avec [0, 1] ou ['No', 'Yes'].
        if len(classes) != 2:
            raise ValueError(f"Classification binaire attendue. Classes trouvées : {classes}")

        positive_column = 1

    y_proba = y_proba_all[:, positive_column]

    # -------------------------------------------------------------------------
    # Classification selon le seuil
    # -------------------------------------------------------------------------
    y_pred = (y_proba >= THRESHOLD).astype(int)

    logger.info(
        {
            "event": "predictions_generated",
            "threshold": THRESHOLD,
            "predictions": len(y_pred),
        }
    )

    return y_pred, y_proba


def evaluate(y_test, y_pred, y_proba):
    """
    Calcule les métriques de classification.
    """

    # -------------------------------------------------------------------------
    # Vérification des dimensions
    # -------------------------------------------------------------------------
    if len(y_test) != len(y_pred):
        raise ValueError(
            "Le nombre de prédictions ne correspond pas "
            "au nombre de valeurs réelles."
        )

    # -------------------------------------------------------------------------
    # Matrice de confusion
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # -------------------------------------------------------------------------
    # Métriques
    # -------------------------------------------------------------------------
    # Sur un petit jeu de test, une seule classe peut être présente :
    # roc_auc_score lève alors une erreur. On renvoie NaN plutôt que d'échouer.
    both_classes = len(set(y_test)) == 2
    scores = {
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "f1": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_test, y_proba)) if both_classes else float("nan"),
        "precision_pluie": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "recall_pluie": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(THRESHOLD),
    }

    return scores


def save_metrics(scores):
    """
    Sauvegarde les métriques dans un fichier JSON.
    """
    METRICS_PATH.parent.mkdir(parents=True,exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=2, ensure_ascii=False)

    logger.info(
        {
            "event": "metrics_saved",
            "path": str(METRICS_PATH),
        }
    )


def save_predictions(y_pred, y_proba):
    """
    Sauvegarde les prédictions et probabilités.
    """
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {
            "prediction": y_pred,
            "probabilite": y_proba,
        }
    )

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    logger.info(
        {
            "event": "predictions_saved",
            "path": str(PREDICTIONS_PATH),
            "rows": len(predictions_df),
        }
    )


def log_dataset_to_mlflow(mlflow) -> None:
    """
    Rattache le dataset au run MLflow courant :
      - le fichier parquet en artefact (stocké sur DagsHub),
      - un dataset MLflow (schéma, empreinte) visible dans l'onglet "Datasets",
      - l'empreinte md5, identique à celle de dvc.lock, pour faire le lien avec DVC.
    """
    if not RAW_DATASET_PATH.exists():
        logger.warning({"event": "raw_dataset_missing", "path": str(RAW_DATASET_PATH)})
        return

    md5 = hashlib.md5(RAW_DATASET_PATH.read_bytes()).hexdigest()
    mlflow.set_tag("dataset_md5", md5)
    mlflow.set_tag("dataset_path", str(RAW_DATASET_PATH))

    try:
        mlflow.log_artifact(str(RAW_DATASET_PATH), artifact_path="dataset")
    except Exception as exc:
        logger.error({"event": "mlflow_dataset_artifact_failed", "error": str(exc)}, exc_info=True)
        return

    try:
        import mlflow.data

        df_raw = pd.read_parquet(RAW_DATASET_PATH)
        dataset = mlflow.data.from_pandas(
            df_raw,
            source=str(RAW_DATASET_PATH),
            name="weather_raw",
            targets=SETTINGS["target"]["column_norm"],
        )
        mlflow.log_input(dataset, context="training")
    except Exception as exc:
        # L'artefact est déjà envoyé : un serveur qui ne gère pas log_input
        # ne doit pas faire échouer l'évaluation.
        logger.warning({"event": "mlflow_log_input_failed", "error": str(exc)})

    logger.info({"event": "dataset_logged_mlflow", "md5": md5, "rows_file": str(RAW_DATASET_PATH)})


def log_mlflow(model, scores, best_params):
    """
    Enregistre le résultat de l'évaluation dans MLflow.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        logger.info(
            {
                "event": "mlflow_disabled",
                "message": (
                    "MLFLOW_TRACKING_URI non défini. "
                    "Aucun logging MLflow."
                ),
            }
        )
        return

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            # ---------------------------------------------------------------
            # Paramètres
            # ---------------------------------------------------------------
            if best_params:
                mlflow.log_params(best_params)

            # ---------------------------------------------------------------
            # Paramètres d'évaluation
            # ---------------------------------------------------------------
            mlflow.log_param("prediction_threshold", THRESHOLD)

            # ---------------------------------------------------------------
            # Métriques
            # ---------------------------------------------------------------
            mlflow.log_metrics(scores)

            # ---------------------------------------------------------------
            # Modèle + enregistrement dans le Model Registry
            # ---------------------------------------------------------------
            # registered_model_name est indispensable : sans lui, le modèle
            # est loggé comme simple artefact du run mais n'apparaît jamais
            # dans le Model Registry, et promote_model.py ne trouve alors
            # jamais de nouvelle version à promouvoir.
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                serialization_format="cloudpickle",
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            # ---------------------------------------------------------------
            # Données d'entraînement
            # ---------------------------------------------------------------
            log_dataset_to_mlflow(mlflow)


        # promote_model.py promeut CETTE version, pas "la plus récente du
        # registre" (qui peut être une autre si ce logging a échoué).
        CANDIDATE_INFO_PATH.write_text(
            json.dumps({
                "run_id": run.info.run_id,
                "version": str(model_info.registered_model_version),
            }),
            encoding="utf-8",
        )

        logger.info(
            {
                "event": "mlflow_logged",
                "tracking_uri": tracking_uri,
                "experiment": MLFLOW_EXPERIMENT_NAME,
                "registered_model_name": REGISTERED_MODEL_NAME,
                "registered_model_version": getattr(
                    model_info, "registered_model_version", None
                ),
            }
        )

    except Exception as exc:
        # Sans enregistrement MLflow, la promotion n'est pas possible :
        # on échoue franchement plutôt que de laisser promote_model
        # comparer une ancienne version.
        logger.error(
            {
                "event": "mlflow_logging_failed",
                "error": str(exc),
            },
            exc_info=True,
        )

        raise


def display_results(scores):
    """
    Affiche les métriques dans la console.
    """

    print()
    print("=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)

    for key, value in scores.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")

        else:
            print(f"{key:20s}: {value}")

    print("=" * 60)
    print("✅ evaluate_model OK")
    print()


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info(
        {
            "event": "evaluate_model_started",
            "model_path": str(MODEL_PATH),
            "dataset_path": str(DATASET_PATH),
        }
    )

    try:
        CANDIDATE_INFO_PATH.unlink(missing_ok=True)

        # ---------------------------------------------------------------------
        # 1. Chargement
        # ---------------------------------------------------------------------
        data, model, metadata, best_params = load_artifacts()

        # ---------------------------------------------------------------------
        # 2. Validation dataset
        # ---------------------------------------------------------------------
        X_test, y_test = validate_dataset(data)

        # ---------------------------------------------------------------------
        # 3. Prédictions
        # ---------------------------------------------------------------------
        y_pred, y_proba = predict(model, X_test)

        # ---------------------------------------------------------------------
        # 4. Évaluation
        # ---------------------------------------------------------------------
        scores = evaluate(y_test, y_pred, y_proba)

        # ---------------------------------------------------------------------
        # 5. Sauvegarde métriques
        # ---------------------------------------------------------------------
        save_metrics(scores)

        # ---------------------------------------------------------------------
        # 6. Sauvegarde prédictions
        # ---------------------------------------------------------------------
        save_predictions(y_pred, y_proba)

        # ---------------------------------------------------------------------
        # 7. MLflow
        # ---------------------------------------------------------------------
        log_mlflow(model, scores, best_params)

        # ---------------------------------------------------------------------
        # 8. Log final
        # ---------------------------------------------------------------------
        logger.info(
            {
                "event": "model_evaluated",
                "scores": scores,
                "metadata": metadata,
            }
        )

        # ---------------------------------------------------------------------
        # 9. Affichage
        # ---------------------------------------------------------------------
        display_results(scores)

    except FileNotFoundError as exc:
        # dataset/modèle absent => une étape en amont a été "skipped" faute
        # de données. Pas de données, pas d'évaluation.
        logger.warning({"event": "evaluate_model_skipped", "reason": str(exc)})
        print(f"⚠️  evaluate_model SKIPPED (artefact manquant) : {exc}")
        sys.exit(SKIP_EXIT_CODE)

    except Exception as exc:
        logger.error(
            {
                "event": "model_evaluation_failed",
                "error": str(exc),
            },
            exc_info=True,
        )

        print(
            f"❌ evaluate_model FAILED: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()