# Prétraitement : on bouche les trous, on met à l'échelle et on encode le texte

import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def main():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    # On sépare les colonnes chiffrées des colonnes texte
    colonnes_num = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    colonnes_cat = X_train.select_dtypes(include=["object"]).columns.tolist()

    # on remplace les valeurs manquantes par la médiane puis on normalise
    traitement_num = Pipeline(steps=[
        ("imputation", SimpleImputer(strategy="median")),
        ("echelle", StandardScaler()),
    ])

    # On remplace les manquants par la valeur la plus fréquente puis encodage du texte
    traitement_cat = Pipeline(steps=[
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocesseur = ColumnTransformer(transformers=[
        ("num", traitement_num, colonnes_num),
        ("cat", traitement_cat, colonnes_cat),
    ])

    # Application des prétraitement sur train et test
    X_train_scaled = preprocesseur.fit_transform(X_train)
    X_test_scaled = preprocesseur.transform(X_test)
    colonnes = preprocesseur.get_feature_names_out()

    pd.DataFrame(X_train_scaled, columns=colonnes).to_csv("data/processed/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=colonnes).to_csv("data/processed/X_test_scaled.csv", index=False)

    # On enregistre les préprocessing pour plus tard et les données futures
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocesseur, "models/preprocessor.pkl")

    print("Prétraitement terminé. train :", X_train_scaled.shape, "| test :", X_test_scaled.shape)


if __name__ == "__main__":
    main()
