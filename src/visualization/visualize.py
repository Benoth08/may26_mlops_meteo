"""
visualize.py — Visualisations EDA pour données météo.
Génère 5 figures statiques dans reports/figures/.
"""

from core.settings import SETTINGS
from core.logger import get_logger

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path


logger = get_logger("visualize_model")

DATA_DIR = Path(SETTINGS["paths"]["processed"])
FIGURES_DIR = Path(SETTINGS["paths"]["figures"])

PREPROCESSED_DATA_PATH = (PROCESSED_DIR / SETTINGS["models"]["preprocessed_data"])


# ── Palette & style global ───────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURES_DIR = Path("FIGURES_DIR")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Noms lisibles des mois en français
MOIS_FR = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
           "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]


# ── Chargement ────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV météo, parse la colonne 'date' et trie chronologiquement."""
    
    if path is None or :
        path = PREPROCESSED_DATA_PATH
       
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    return df


# ── 1. Série temporelle (température + précipitations sur 2 axes) ─────────────
def plot_temperature_series(df: pd.DataFrame):
    """
    Ligne de température (axe gauche, rouge) et barres de précipitations
    (axe droit, bleu) sur la même figure — lecture instantanée du climat.
    """
    fig, ax1 = plt.subplots(figsize=(14, 4))

    # Température : courbe lissée sur 7 jours pour lisser le bruit journalier
    temp_smooth = df["temperature"].rolling(7, center=True).mean()
    ax1.plot(df.index, df["temperature"], color="tomato", alpha=0.3, linewidth=0.6)
    ax1.plot(df.index, temp_smooth, color="tomato", linewidth=1.8, label="Temp. (moy. 7j)")
    ax1.set_ylabel("Température (°C)", color="tomato")
    ax1.tick_params(axis="y", labelcolor="tomato")

    # Précipitations sur axe droit si la colonne existe
    if "precipitation" in df.columns:
        ax2 = ax1.twinx()
        ax2.bar(df.index, df["precipitation"], color="steelblue", alpha=0.35, width=1, label="Précip.")
        ax2.set_ylabel("Précipitations (mm)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")

    # Format de date lisible sur l'axe X
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    ax1.set_title("Série temporelle — Température & Précipitations")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "temperature_series.png", dpi=150)
    plt.close(fig)


# ── 2. Distributions de toutes les variables numériques ──────────────────────
def plot_distributions(df: pd.DataFrame):
    """
    Histogramme + courbe KDE pour chaque variable numérique.
    KDE permet de détecter bimodalité / asymétrie au premier coup d'œil.
    """
    num_cols = df.select_dtypes("number").columns.tolist()
    n = len(num_cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, num_cols):
        # kde=True superpose la densité estimée à l'histogramme
        sns.histplot(df[col].dropna(), bins=35, kde=True, ax=ax,
                     color="steelblue", edgecolor="white", alpha=0.75)
        ax.set(title=col, xlabel="", ylabel="Fréquence")

    fig.suptitle("Distribution des variables météo", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 3. Heatmap de corrélation ─────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Corrélations de Pearson entre toutes les variables numériques.
    Triangle inférieur uniquement pour éviter la redondance visuelle.
    """
    corr = df.select_dtypes("number").corr()

    # Masque du triangle supérieur (redondant avec le bas)
    import numpy as np
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size": 9}, ax=ax)
    ax.set_title("Matrice de corrélation (Pearson)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


# ── 4. Boxplot mensuel de la température ─────────────────────────────────────
def plot_monthly_boxplot(df: pd.DataFrame):
    """
    Boxplot de la température pour chaque mois : saisonnalité et outliers visibles.
    """
    df_copy = df[["temperature"]].copy()
    df_copy["mois"] = df_copy.index.month  # entier 1–12

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df_copy, x="mois", y="temperature",
                palette="coolwarm", ax=ax, flierprops={"marker": ".", "alpha": 0.4})

    ax.set_xticks(range(12))
    ax.set_xticklabels(MOIS_FR)
    ax.set(title="Distribution mensuelle de la température",
           xlabel="Mois", ylabel="Température (°C)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monthly_boxplot.png", dpi=150)
    plt.close(fig)


# ── 5. Heatmap calendrier (température par jour/mois) ────────────────────────
def plot_calendar_heatmap(df: pd.DataFrame):
    """
    Pivot année × mois : repère les mois chauds/froids sur plusieurs années
    en un seul coup d'œil — typique des dashboards météo.
    """
    df_copy = df[["temperature"]].copy()
    df_copy["annee"] = df_copy.index.year
    df_copy["mois"] = df_copy.index.month

    # Moyenne mensuelle par année
    pivot = df_copy.groupby(["annee", "mois"])["temperature"].mean().unstack()
    pivot.columns = MOIS_FR  # noms lisibles en colonnes

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.7)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlBu_r",
                linewidths=0.4, ax=ax, cbar_kws={"label": "°C"})
    ax.set(title="Température moyenne mensuelle par année",
           xlabel="Mois", ylabel="Année")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calendar_heatmap.png", dpi=150)
    plt.close(fig)


# ── Pipeline principal ────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    logger.info({"event": "visualize / loading data"})
        
    df = load_data()

    plot_temperature_series(df)
    logger.info({"event": "visualize / Série temporelle : OK"})
    print("✓ Série temporelle")

    plot_distributions(df)
    logger.info({"event": "visualize / Distributions : OK"})
    print("✓ Distributions")

    plot_correlation_heatmap(df)
    logger.info({"event": "visualize / Heatmap corrélation : OK"})
    print("✓ Heatmap corrélation")

    plot_monthly_boxplot(df)
    logger.info({"event": "visualize / Boxplot mensuel : OK"})
    print("✓ Boxplot mensuel")

    plot_calendar_heatmap(df)
    logger.info({"event": "visualize / Calendrier heatmap : OK"})
    print("✓ Calendrier heatmap")

    logger.info({"event": "visualize / Figures sauvegardées → {FIGURES_DIR.resolve()}"})
    print(f"\nFigures sauvegardées → {FIGURES_DIR.resolve()}")
