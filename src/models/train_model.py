# Entraînement final : on réentraîne le modèle avec les meilleurs réglages trouvés

import pandas as pd
import joblib
from pathlib import Path

from lightgbm import LGBMClassifier


def main():
    X_train = pd.read_csv("data/processed/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    # On récupère les hyperparamètres gagnants de l'étape précédente.
    best_params = joblib.load("models/best_params.pkl")

    # Le même modèle, mais cette fois avec les bons réglages, entraîné sur tout le train.
    modele = LGBMClassifier(class_weight="balanced", verbosity=-1,random_state=42, n_jobs=-1, **best_params)
    modele.fit(X_train, y_train)

    # On sauvegarde le modèle entraîné : c'est lui que l'API ira charger.
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, "models/model.pkl")

    print("Modèle entraîné et sauvegardé dans models/model.pkl")


if __name__ == "__main__":
    main()
