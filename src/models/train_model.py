"""Étape 3 — Entraîne le pipeline complet (préprocesseur + LightGBM) avec les
meilleurs hyperparamètres, sur tout le train."""
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


def main():
    data = joblib.load("data/processed/dataset.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    best_params = joblib.load("models/best_params.joblib")

    lgbm_params = {k.replace("model__", ""): v for k, v in best_params.items()}

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbosity=-1, **lgbm_params)),
    ])
    pipe.fit(data["X_train"], data["y_train"])

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, "models/model.joblib")

    print("✅ train_model OK -> models/model.joblib")
    print("   Hyperparamètres :", lgbm_params)


if __name__ == "__main__":
    main()
