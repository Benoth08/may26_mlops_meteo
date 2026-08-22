"""Tests unitaires des fonctions déterministes de build_features.

Depuis la restructuration de src/ (core, features, ...), toutes les
fonctions de build_features opèrent sur des noms de colonnes normalisés
(snake_case, ex. "min_temp", "rain_tomorrow") : elles appellent
core.metadata.normalize_column_name en interne et lisent/écrivent ces
colonnes normalisées. `raw_df` reproduit le CSV source (noms métier
CamelCase) et sert au test end-to-end de prepare_data ; `norm_df` est
son équivalent déjà normalisé (via bf.normalize_data) et sert aux tests
unitaires des étapes intermédiaires.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf


@pytest.fixture
def raw_df():
    n = 12
    rng = np.random.default_rng(0)
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Location": ["Albury", "Sydney"] * (n // 2),
        "MinTemp": rng.uniform(5, 15, n),
        "MaxTemp": rng.uniform(20, 35, n),
        "Rainfall": rng.uniform(0, 10, n),
        "Evaporation": rng.uniform(0, 10, n),
        "Sunshine": rng.uniform(0, 12, n),
        "WindGustDir": [dirs[i % len(dirs)] for i in range(n)],
        "WindGustSpeed": rng.uniform(20, 60, n),
        "WindDir9am": [dirs[i % len(dirs)] for i in range(n)],
        "WindDir3pm": [dirs[(i + 1) % len(dirs)] for i in range(n)],
        "WindSpeed9am": rng.uniform(0, 30, n),
        "WindSpeed3pm": rng.uniform(0, 30, n),
        "Humidity9am": rng.uniform(30, 100, n),
        "Humidity3pm": rng.uniform(20, 90, n),
        "Pressure9am": rng.uniform(1000, 1030, n),
        "Pressure3pm": rng.uniform(1000, 1030, n),
        "Cloud9am": rng.integers(0, 9, n).astype(float),
        "Cloud3pm": rng.integers(0, 9, n).astype(float),
        "Temp9am": rng.uniform(10, 25, n),
        "Temp3pm": rng.uniform(15, 30, n),
        "RainToday": ["No", "Yes"] * (n // 2),
        "RainTomorrow": ["No", "Yes"] * (n // 2),
    })


@pytest.fixture
def norm_df(raw_df):
    return bf.normalize_data(raw_df.copy())


def test_validate_schema_ok(norm_df):
    bf.validate_schema(norm_df)  # ne doit pas lever


def test_validate_schema_missing(norm_df):
    with pytest.raises(ValueError):
        bf.validate_schema(norm_df.drop(columns=["min_temp"]))


def test_clean_target_maps_and_drops(norm_df):
    df = norm_df.copy()
    df.loc[0, "rain_tomorrow"] = np.nan
    out = bf.clean_target(df)
    assert pd.api.types.is_integer_dtype(out["rain_tomorrow"])
    assert set(out["rain_tomorrow"].unique()).issubset({0, 1})
    assert len(out) == len(df) - 1


def test_encode_rain_today(norm_df):
    out = bf.encode_rain_today(norm_df)
    assert set(out["rain_today"].dropna().unique()).issubset({0, 1})


def test_temporal_and_cyclical(norm_df):
    df = bf.add_temporal_features(bf.parse_date_column(norm_df))
    assert {"year", "month", "day", "day_of_year"}.issubset(df.columns)
    df = bf.add_cyclical_features(df)
    for c in ["month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]:
        assert df[c].between(-1, 1).all()


def test_encode_wind_directions(norm_df):
    out = bf.encode_wind_directions(norm_df)
    for col in bf.WIND_DIRECTION_COLUMNS:
        assert col not in out.columns
        assert f"{col}_sin" in out.columns and f"{col}_cos" in out.columns


def test_add_weather_features(norm_df):
    out = bf.add_weather_features(norm_df)
    assert "temp_range" in out.columns
    np.testing.assert_allclose(out["temp_range"], norm_df["max_temp"] - norm_df["min_temp"])


def test_split_features_target(norm_df):
    X, y = bf.split_features_target(bf.clean_target(norm_df))
    assert "rain_tomorrow" not in X.columns
    assert len(X) == len(y)


def test_identify_high_missing(norm_df):
    df = norm_df.copy()
    df["sunshine"] = np.nan
    assert "sunshine" in bf.identify_high_missing_columns(df, threshold=0.3)


def test_build_preprocessor_smoke():
    pre = bf.build_preprocessor(["min_temp", "max_temp"], ["location"])
    assert hasattr(pre, "fit")


def test_prepare_data_end_to_end(tmp_path, raw_df):
    csv = tmp_path / "w.csv"
    raw_df.to_csv(csv, index=False)
    data = bf.prepare_data(source="csv", data_path=str(csv),
                           split_strategy="random", save_report=False)
    assert {"X_train", "X_test", "preprocessor"}.issubset(data)
    assert len(data["X_train"]) + len(data["X_test"]) == len(raw_df)
