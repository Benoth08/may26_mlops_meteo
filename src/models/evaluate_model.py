"""Étape 4 — Évalue le modèle sur le jeu de test mis de côté."""
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
)


def main():
    data = joblib.load("data/processed/dataset.joblib")
    model = joblib.load("models/model.joblib")

    X_test, y_test = data["X_test"], data["y_test"]
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision_pluie": precision_score(y_test, y_pred),
        "recall_pluie": recall_score(y_test, y_pred),
    }

    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=float)
    pd.DataFrame({"prediction": y_pred, "probabilite": y_proba}).to_csv(
        "data/predictions.csv", index=False)

    print("✅ evaluate_model OK")
    for k, v in scores.items():
        print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
