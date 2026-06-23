# Recherche des hyperparamètres crossvalin

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from lightgbm import LGBMClassifier


def main():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    # Equilibrage des classes, car il pleut moins souvent donc classe unbalanced
    modele = LGBMClassifier(class_weight="balanced", verbosity=-1, random_state=42, n_jobs=-1)

    # Hyperparamètres à tester
    grille = {
        "n_estimators": [100, 200],
        "num_leaves": [31, 63],
    }

    # Validation croisée pour série temporelle
    cv = TimeSeriesSplit(n_splits=3)
    recherche = GridSearchCV(modele, grille, cv=cv, scoring="f1", n_jobs=-1)
    recherche.fit(X_train, y_train)

    # On garde le best modèle
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(recherche.best_params_, "models/best_params.pkl")

    print("Meilleurs paramètres :", recherche.best_params_)
    print("F1 (validation croisée) :", round(recherche.best_score_, 4))


if __name__ == "__main__":
    main()
