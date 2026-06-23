# Évaluation : on mesure les performances du modèle sur le test mis de côté.

import json
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, precision_score, recall_score,
)


def main():
    X_test = pd.read_csv("data/processed/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    modele = joblib.load("models/model.pkl")

    # Pour chaque jour : la prédiction (0 ou 1) et la probabilité de pluie qui va avec.
    y_pred = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    # On garde une trace des prédictions, ligne par ligne.
    Path("data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": y_pred, "probabilite": y_proba}).to_csv("data/predictions.csv", index=False)

    # On calcule plusieurs métriques pour avoir une vue d'ensemble
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision_pluie": precision_score(y_test, y_pred),
        "recall_pluie": recall_score(y_test, y_pred),
    }

    # On écrit les scores dans un fichier JSON
    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print("Évaluation terminée.")
    for nom, valeur in scores.items():
        print(f"  {nom} : {valeur:.4f}")


if __name__ == "__main__":
    main()
