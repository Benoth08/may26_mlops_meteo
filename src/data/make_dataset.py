"""Étape 1 — Prépare les données via build_features (branche preprocessing).
Produit les splits train/test temporels + le préprocesseur NON entraîné."""
import joblib
from pathlib import Path

from src.features.build_features import prepare_data

DATA_PATH = "data/raw/weatherAUS.csv"


def main():
    data = prepare_data(
        source="csv",
        data_path=DATA_PATH,
        split_strategy="temporal",
        save_report=True,
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {k: data[k] for k in ("X_train", "X_test", "y_train", "y_test")},
        "data/processed/dataset.joblib",
    )
    joblib.dump(data["preprocessor"], "models/preprocessor.joblib")

    print("✅ make_dataset OK")
    print("   X_train :", data["X_train"].shape, "| X_test :", data["X_test"].shape)


if __name__ == "__main__":
    main()
