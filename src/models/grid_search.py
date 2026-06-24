"""Étape 2 — GridSearch (TimeSeriesSplit). Pipeline = préprocesseur + LightGBM,
re-fitté sur le train de chaque pli -> aucune fuite."""
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier


def main():
    data = joblib.load("data/processed/dataset.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=1, verbosity=-1)),
    ])

    grid = {"model__n_estimators": [100, 200], "model__num_leaves": [31, 63]}
    cv = TimeSeriesSplit(n_splits=3)
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    search.fit(data["X_train"], data["y_train"])

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_params_, "models/best_params.joblib")

    print("✅ grid_search OK")
    print("   Meilleurs paramètres :", search.best_params_)
    print("   F1 (CV) :", round(float(search.best_score_), 4))


if __name__ == "__main__":
    main()
