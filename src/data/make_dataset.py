import pandas as pd
from pathlib import Path


def main():
    df = pd.read_csv("data/raw/weatherAUS.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    df = df.dropna(subset=["RainTomorrow"])
    X = df.drop(columns=["RainTomorrow", "Date"])
    y = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    splitage = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:splitage], X.iloc[splitage:]
    y_train, y_test = y.iloc[:splitage], y.iloc[splitage:]

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print("Split terminé. train :", X_train.shape, "| test :", X_test.shape)


if __name__ == "__main__":
    main()
