"""Dashboard Weather MLOps — Vue d'ensemble + Prétraitement + EDA + Prédiction."""
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import build_features as bf

DATA_PATH = "data/raw/weatherAUS.csv"
MODEL_PATH = "models/model.joblib" if Path("models/model.joblib").exists() else "models/model.pkl"

NUMERIC_COLS = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
                "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
                "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]
WIND_DIRS = list(bf.COMPASS_DEGREES.keys())

st.set_page_config(page_title="Weather MLOps", page_icon="🌧️", layout="wide")


# ── Chargements (mis en cache) ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_scores():
    p = Path("metrics/scores.json")
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def run_preprocessing():
    return bf.prepare_data(source="csv", data_path=DATA_PATH,
                           split_strategy="temporal", save_report=False)


def featurize(df_raw):
    """Rejoue le feature engineering déterministe de build_features (sans la cible)."""
    df = df_raw.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = bf.encode_rain_today(df)
    df = bf.parse_date_column(df)
    df = bf.add_temporal_features(df)
    df = bf.add_cyclical_features(df)
    df = bf.encode_wind_directions(df)
    df = bf.add_weather_features(df)
    df = bf.drop_unused_columns(df)
    drop = [bf.TARGET] + bf.TECHNICAL_COLUMNS
    return df.drop(columns=[c for c in drop if c in df.columns])


data = load_data()
model = load_model()
scores = load_scores()

st.sidebar.title("🌧️ Weather MLOps")
page = st.sidebar.radio(
    "Navigation",
    ["Vue d'ensemble", "Prétraitement", "Visualisation (EDA)", "Prédiction", "Prédiction par lot"],
)

# ───────────────────────────── VUE D'ENSEMBLE ────────────────────────────────
if page == "Vue d'ensemble":
    st.title("Vue d'ensemble")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(data):,}")
    c2.metric("Stations", data["Location"].nunique())
    c3.metric("Pluie demain", f"{(data['RainTomorrow'] == 'Yes').mean() * 100:.1f}%")
    c4.metric("Période", f"{data['Date'].min().year}–{data['Date'].max().year}")

    if scores:
        st.subheader("Performance du modèle (jeu de test)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("ROC-AUC", f"{scores['roc_auc']:.3f}")
        s2.metric("F1", f"{scores['f1']:.3f}")
        s3.metric("Recall pluie", f"{scores['recall_pluie']:.3f}")
        s4.metric("Précision pluie", f"{scores['precision_pluie']:.3f}")

    st.subheader("Aperçu des données")
    st.dataframe(data.head(50), use_container_width=True)

# ───────────────────────────── PRÉTRAITEMENT ─────────────────────────────────
elif page == "Prétraitement":
    st.title("🧪 Prétraitement (build_features)")
    st.caption("Pipeline de la branche preprocessing : du brut aux features, sans fuite de données.")

    st.subheader("Étapes du pipeline")
    st.markdown(
        "1. **Validation du schéma**  ·  2. **Rapport des valeurs manquantes**  ·  "
        "3. **Colonnes > 30 % manquantes** (conservées + imputées)\n"
        "4. **Nettoyage de la cible** (Yes/No → 1/0)  ·  5. **Encodage RainToday**  ·  "
        "6. **Parsing de la date**\n"
        "7. **Features temporelles**  ·  8. **Encodage cyclique** (mois/jour)  ·  "
        "9. **Directions de vent cycliques**\n"
        "10. **Features météo** (TempRange, HumidityDrop, …)  ·  11. **Drop colonnes inutiles**  ·  "
        "12. **Split X / y**\n"
        "13. **Split train/test temporel**  ·  14. **Préprocesseur** "
        "(IterativeImputer + RobustScaler + TargetEncoder)"
    )

    with st.expander("🔬 Voir les transformations étape par étape (échantillon de 200 lignes)"):
        s = data.dropna(subset=["RainTomorrow"]).head(200).copy()
        s["Date"] = s["Date"].astype(str)
        rows = [("0. données brutes", s.shape[1])]
        s = bf.clean_target(s);           rows.append(("4. clean_target", s.shape[1]))
        s = bf.encode_rain_today(s);      rows.append(("5. encode_rain_today", s.shape[1]))
        s = bf.parse_date_column(s);      rows.append(("6. parse_date", s.shape[1]))
        s = bf.add_temporal_features(s);  rows.append(("7. temporel", s.shape[1]))
        s = bf.add_cyclical_features(s);  rows.append(("8. cyclique", s.shape[1]))
        s = bf.encode_wind_directions(s); rows.append(("9. vent (sin/cos)", s.shape[1]))
        s = bf.add_weather_features(s);   rows.append(("10. features météo", s.shape[1]))
        st.dataframe(pd.DataFrame(rows, columns=["Étape", "Nb colonnes"]),
                     use_container_width=True, hide_index=True)
        st.caption("Aperçu après feature engineering :")
        st.dataframe(s.head(), use_container_width=True)

    st.subheader("Résultats sur le jeu complet")
    with st.spinner("Exécution du prétraitement..."):
        prep = run_preprocessing()

    c1, c2, c3 = st.columns(3)
    c1.metric("Features finales", prep["X_train"].shape[1])
    c2.metric("Train", f"{len(prep['X_train']):,}")
    c3.metric("Test", f"{len(prep['X_test']):,}")

    st.markdown("**Rapport des valeurs manquantes**")
    rep = prep["missing_report"]
    st.dataframe(rep, use_container_width=True, height=280)
    miss = rep[rep["missing_rate_pct"] > 0].set_index("column")["missing_rate_pct"]
    if not miss.empty:
        st.bar_chart(miss)

    st.markdown(f"**Colonnes > 30 % manquantes** : `{prep['high_missing_columns'] or 'aucune'}`")

    a, b = st.columns(2)
    with a:
        st.markdown(f"**Numériques** ({len(prep['numeric_features'])})")
        st.write(prep["numeric_features"])
    with b:
        st.markdown(f"**Catégorielles** ({len(prep['categorical_features'])})")
        st.write(prep["categorical_features"])

    st.markdown("**Distribution de la cible (train vs test)**")
    dist = pd.DataFrame({
        "train": prep["y_train"].value_counts(normalize=True),
        "test": prep["y_test"].value_counts(normalize=True),
    }).fillna(0)
    st.bar_chart(dist)

    st.markdown("**Aperçu des features (X_train)**")
    st.dataframe(prep["X_train"].head(), use_container_width=True)

    st.markdown("**Structure du préprocesseur**")
    st.code(str(prep["preprocessor"]))

    if st.button("⚙️ Tester le préprocesseur (fit sur 3 000 lignes)"):
        with st.spinner("fit_transform sur un échantillon..."):
            pre = bf.build_preprocessor(prep["numeric_features"], prep["categorical_features"])
            Xt = pre.fit_transform(prep["X_train"].head(3000), prep["y_train"].head(3000))
        st.success(f"Sortie du préprocesseur : **{Xt.shape}** (lignes × features encodées) ✅")

# ───────────────────────────── EDA ───────────────────────────────────────────
elif page == "Visualisation (EDA)":
    st.title("Exploration des données")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Répartition RainTomorrow")
        st.bar_chart(data["RainTomorrow"].value_counts(dropna=False))
    with col_right:
        st.subheader("Taux de pluie par station (top 15)")
        by_loc = (data.groupby("Location")["RainTomorrow"]
                  .apply(lambda s: (s == "Yes").mean())
                  .sort_values(ascending=False).head(15))
        st.bar_chart(by_loc)

    st.subheader("Distribution d'une variable")
    col = st.selectbox("Variable", NUMERIC_COLS, index=NUMERIC_COLS.index("Humidity3pm"))
    fig, ax = plt.subplots(figsize=(9, 3))
    sns.histplot(data[col].dropna(), bins=40, kde=True, ax=ax, color="steelblue")
    st.pyplot(fig)

    st.subheader("Corrélations entre variables numériques")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.heatmap(data[NUMERIC_COLS].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax2, annot_kws={"size": 7})
    st.pyplot(fig2)

    st.subheader("Évolution temporelle (par station)")
    loc = st.selectbox("Station", sorted(data["Location"].dropna().unique()))
    sub = data[data["Location"] == loc].dropna(subset=["Date"]).set_index("Date").sort_index()
    monthly = sub[["MaxTemp", "MinTemp", "Rainfall"]].resample("ME").mean()
    st.line_chart(monthly[["MaxTemp", "MinTemp"]])
    st.bar_chart(monthly["Rainfall"])

    st.subheader("Valeurs manquantes (%)")
    miss = (data.isna().mean() * 100).sort_values(ascending=False)
    st.bar_chart(miss[miss > 0])

# ───────────────────────────── PRÉDICTION ────────────────────────────────────
elif page == "Prédiction":
    st.title("Prédire la pluie de demain")
    st.caption("Renseigne les observations du jour → le modèle prédit s'il pleuvra demain.")

    if "sample" not in st.session_state:
        st.session_state.sample = data.dropna(subset=["Location"]).sample(1).iloc[0]
    if st.button("🎲 Charger un exemple réel"):
        st.session_state.sample = data.dropna(subset=["Location"]).sample(1).iloc[0]
    ex = st.session_state.sample

    def nd(c, fb):
        v = ex.get(c)
        return float(v) if pd.notna(v) else fb

    def diridx(c):
        return WIND_DIRS.index(ex[c]) if ex.get(c) in WIND_DIRS else 0

    locs = sorted(data["Location"].dropna().unique())
    c1, c2, c3 = st.columns(3)
    with c1:
        location = st.selectbox("Station", locs,
                                index=locs.index(ex["Location"]) if ex["Location"] in locs else 0)
        date = st.date_input("Date", value=pd.Timestamp("2025-06-24"))
        rain_today = st.selectbox("Pluie aujourd'hui ?", ["No", "Yes"],
                                  index=1 if ex.get("RainToday") == "Yes" else 0)
        mintemp = st.number_input("MinTemp (°C)", value=nd("MinTemp", 12.0))
        maxtemp = st.number_input("MaxTemp (°C)", value=nd("MaxTemp", 23.0))
        rainfall = st.number_input("Rainfall (mm)", value=nd("Rainfall", 0.0))
        evaporation = st.number_input("Evaporation (mm)", value=nd("Evaporation", 5.0))
    with c2:
        sunshine = st.number_input("Sunshine (h)", value=nd("Sunshine", 8.0))
        windgustdir = st.selectbox("WindGustDir", WIND_DIRS, index=diridx("WindGustDir"))
        windgustspeed = st.number_input("WindGustSpeed (km/h)", value=nd("WindGustSpeed", 40.0))
        windspeed9 = st.number_input("WindSpeed9am", value=nd("WindSpeed9am", 15.0))
        windspeed3 = st.number_input("WindSpeed3pm", value=nd("WindSpeed3pm", 18.0))
        humidity9 = st.number_input("Humidity9am (%)", value=nd("Humidity9am", 70.0))
        humidity3 = st.number_input("Humidity3pm (%)", value=nd("Humidity3pm", 50.0))
    with c3:
        pressure9 = st.number_input("Pressure9am (hPa)", value=nd("Pressure9am", 1015.0))
        pressure3 = st.number_input("Pressure3pm (hPa)", value=nd("Pressure3pm", 1013.0))
        cloud9 = st.number_input("Cloud9am (octas)", value=nd("Cloud9am", 4.0))
        cloud3 = st.number_input("Cloud3pm (octas)", value=nd("Cloud3pm", 4.0))
        temp9 = st.number_input("Temp9am (°C)", value=nd("Temp9am", 16.0))
        temp3 = st.number_input("Temp3pm (°C)", value=nd("Temp3pm", 21.0))
        winddir9 = st.selectbox("WindDir9am", WIND_DIRS, index=diridx("WindDir9am"))
        winddir3 = st.selectbox("WindDir3pm", WIND_DIRS, index=diridx("WindDir3pm"))

    if st.button("🔮 Prédire", type="primary"):
        row = {
            "Date": str(date), "Location": location, "MinTemp": mintemp, "MaxTemp": maxtemp,
            "Rainfall": rainfall, "Evaporation": evaporation, "Sunshine": sunshine,
            "WindGustDir": windgustdir, "WindGustSpeed": windgustspeed,
            "WindDir9am": winddir9, "WindDir3pm": winddir3, "WindSpeed9am": windspeed9,
            "WindSpeed3pm": windspeed3, "Humidity9am": humidity9, "Humidity3pm": humidity3,
            "Pressure9am": pressure9, "Pressure3pm": pressure3, "Cloud9am": cloud9,
            "Cloud3pm": cloud3, "Temp9am": temp9, "Temp3pm": temp3, "RainToday": rain_today,
        }
        X = featurize(pd.DataFrame([row]))
        proba = float(model.predict_proba(X)[:, 1][0])
        col_a, col_b = st.columns(2)
        col_a.metric("Demain", "Pluie 🌧️" if proba >= 0.5 else "Pas de pluie ☀️")
        col_b.metric("Probabilité de pluie", f"{proba * 100:.1f}%")
        st.progress(proba)
        if pd.notna(ex.get("RainTomorrow")):
            st.caption(f"(Exemple chargé — valeur réelle : RainTomorrow = **{ex['RainTomorrow']}**)")

# ───────────────────────────── PRÉDICTION PAR LOT ────────────────────────────
elif page == "Prédiction par lot":
    st.title("Prédiction par lot (CSV)")
    st.caption("Dépose un CSV aux colonnes weatherAUS — une prédiction par ligne.")
    up = st.file_uploader("Fichier CSV", type="csv")
    if up:
        raw = pd.read_csv(up)
        X = featurize(raw)
        proba = model.predict_proba(X)[:, 1]
        res = pd.DataFrame({
            "Date": raw.loc[X.index, "Date"].values,
            "Location": raw.loc[X.index, "Location"].values,
            "RainTomorrow_pred": np.where(proba >= 0.5, "Yes", "No"),
            "Probabilité": proba.round(4),
        })
        st.success(f"{len(res)} prédictions générées")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Télécharger", res.to_csv(index=False),
                           "predictions.csv", "text/csv")
