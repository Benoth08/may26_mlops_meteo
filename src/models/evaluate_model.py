"""Étape 4 — Évalue le modèle sur le jeu de test mis de côté."""
import os
import sys
import json
import joblib

from pathlib import Path

from core.settings import SETTINGS
from core.logger import get_logger
 
for _env_key, _env_val in SETTINGS["threads"].items():
    os.environ.setdefault(_env_key, str(_env_val))

from sklearn.experimental import enable_iterative_imputer  # noqa: F401

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
)

logger = get_logger("evaluate_model")

MODELS_DIR = Path(SETTINGS["paths"]["models"])
METRICS_DIR = Path(SETTINGS["paths"]["metrics"])
PROCESSED_DIR = Path(SETTINGS["paths"]["processed"])
DATA_DIR = Path(SETTINGS["paths"]["data"])

DATASET_PATH = (PROCESSED_DIR / SETTINGS["models"]["dataset"])
MODEL_PATH = (MODELS_DIR / SETTINGS["models"]["model"])
METRICS_PATH = (METRICS_DIR / SETTINGS["models"]["metrics"])
PREDICTIONS_PATH = (DATA_DIR / SETTINGS["models"]["predictions"])
BEST_PARAMS_PATH = (MODELS_DIR / SETTINGS["models"]["best_params"])


def main():
    
    logger.info({"event": "loading_evaluate_model", "model_path": str(MODEL_PATH)})
    
    try:
        data = joblib.load(DATASET_PATH)
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            model = artifact["pipeline"]
            metadata = artifact.get("metadata", {})
        else:
            model = artifact
            metadata = {}
        best_params = joblib.load(BEST_PARAMS_PATH)
    except FileNotFoundError as e:
        logger.error({"event": "artifact_not_found", "error": str(e)}, exc_info=True)
        sys.exit(1)

    X_test, y_test = data["X_test"], data["y_test"]
    
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        logger.error({"event": "prediction_failed", "error": str(e)}, exc_info=True)
        sys.exit(1)

    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision_pluie": precision_score(y_test, y_pred),
        "recall_pluie": recall_score(y_test, y_pred),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(scores, f, indent=2, default=float)
 
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": y_pred, "probabilite": y_proba}).to_csv(
        PREDICTIONS_PATH, index=False)

    # Suivi MLflow (DagsHub) : desactive si MLFLOW_TRACKING_URI n'est pas defini,
    # pour ne pas casser `dvc repro` sur une machine non configuree.
    if os.environ.get("MLFLOW_TRACKING_URI"):
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("weather-rain-prediction")
        with mlflow.start_run():
            mlflow.log_params(best_params)
            mlflow.log_metrics(scores)
            # cloudpickle : le format par defaut (skops) ne reconnait pas
            # encore les objets LightGBM (UntrustedTypesFoundException).
            mlflow.sklearn.log_model(
                model, "model", serialization_format="cloudpickle")

    logger.info({"event": "model_evaluated", "scores": scores})
    
    print("✅ evaluate_model OK")
    for k, v in scores.items():
        print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
