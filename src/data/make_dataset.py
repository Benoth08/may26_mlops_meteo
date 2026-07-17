# Préparation des données : on charge le CSV brut et on le coupe en train / test.

import pandas as pd
from pathlib import Path


def main():
    # On lit les données brutes et on fait une mise en forme de la colonne date en vrai date
    df = pd.read_csv("data/raw/weatherAUS.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Etape de rangement des dates au cours du temps
    df = df.sort_values("Date").reset_index(drop=True)

    # Suprression des données manquantes et normalisation des données categorielles
    df = df.dropna(subset=["RainTomorrow"])
    X = df.drop(columns=["RainTomorrow", "Date"])
    y = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    # Split 80/20
    splitage = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:splitage], X.iloc[splitage:]
    y_train, y_test = y.iloc[:splitage], y.iloc[splitage:]

    # On enregistre les 4 splits
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Split terminé. train :", X_train.shape, "| test :", X_test.shape)


if __name__ == "__main__":
    main()
