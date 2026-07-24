-- ===========================================================================
-- WeatherAUS — Table des features ML (weather_features_ml)
-- ---------------------------------------------------------------------------
-- Contrairement à weather_data_clean, le nombre et le nom des colonnes ne
-- sont pas fixes ici : après IterativeImputer / RobustScaler / OneHotEncoder
-- / TargetEncoder (build_preprocessor dans build_features.py), le nombre de
-- colonnes dépend des catégories présentes dans le jeu d'entraînement
-- (ex : une colonne one-hot par ville rencontrée). Un schéma SQL rigide
-- casserait à chaque nouvel entraînement. On stocke donc les features
-- transformées en JSONB : 1 ligne = 1 observation, 1 clé JSON = 1 feature.
--
-- Alimentée par normalize_data.py (fonction save_ml_data_to_postgres),
-- appelée après le traitement ML (fit_transform / transform du préprocesseur).
-- ===========================================================================

CREATE TABLE IF NOT EXISTS weather_features_ml (
    id BIGSERIAL PRIMARY KEY,

    run_id TEXT NOT NULL,
    dataset_split TEXT NOT NULL CHECK (dataset_split IN ('train', 'test')),
    row_index INTEGER NOT NULL,

    -- Une clé JSON par feature transformée (ex: "num__min_temp", "cat__location_Sydney", ...)
    features JSONB NOT NULL,

    -- Cible (rain_tomorrow encodée 0/1), nullable si non disponible au moment du stockage
    target SMALLINT,

    date_import TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_weather_features_ml_run_split
    ON weather_features_ml (run_id, dataset_split);

CREATE INDEX IF NOT EXISTS ix_weather_features_ml_features_gin
    ON weather_features_ml USING GIN (features);
