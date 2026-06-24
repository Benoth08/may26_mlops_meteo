"""Chaîne complète : prétraitement (branche preprocessing) -> GridSearch -> train -> eval."""
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
)
from lightgbm import LGBMClassifier

from src.features.build_features import prepare_data


def main():
    # 1. Prétraitement développé (branche preprocessing) + split temporel
    data = prepare_data(
        source="csv",
        data_path="data/raw/weatherAUS.csv",
        split_strategy="temporal",
        save_report=True,
    )
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    preprocessor = data["preprocessor"]  # ColumnTransformer NON fitté

    # 2. Pipeline = préprocesseur + modèle (re-fit sur chaque pli -> aucune fuite)
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, verbosity=-1)),
    ])

    # 3. GridSearch en validation temporelle
    grid = {
        "model__n_estimators": [100, 200],
        "model__num_leaves": [31, 63],
    }
    cv = TimeSeriesSplit(n_splits=3)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print("Meilleurs paramètres :", search.best_params_)
    print("F1 (validation croisée) :", round(float(search.best_score_), 4))

    best = search.best_estimator_

    # 4. Évaluation sur le test mis de côté
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision_pluie": precision_score(y_test, y_pred),
        "recall_pluie": recall_score(y_test, y_pred),
        "best_params": search.best_params_,
        "cv_f1": float(search.best_score_),
    }

    # 5. Sauvegardes
    for d in ("models", "metrics", "data"):
        Path(d).mkdir(parents=True, exist_ok=True)
    joblib.dump(best, "models/model.pkl")  # pipeline COMPLET (prep + modèle)
    pd.DataFrame({"prediction": y_pred, "probabilite": y_proba}).to_csv(
        "data/predictions.csv", index=False)
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=float)

    print("\n=== Scores (jeu de test) ===")
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print("\nModèle complet sauvegardé -> models/model.pkl")


if __name__ == "__main__":
    main()
