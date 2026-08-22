-- ===========================================================================
-- WeatherAUS — Table des données nettoyées / typées (weather_data_clean)
-- ---------------------------------------------------------------------------
-- Alimentée par le script Python `load_clean_data.py`, en amont de
-- build_features.py (source="postgres"), après normalisation des colonnes
-- (normalize_data), validation du schéma (validate_schema) et conversion
-- des types (convert_types). Aucune feature engineering / encodage ML ici.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS weather_data_clean (
    id BIGSERIAL PRIMARY KEY,

    -- Colonnes métier (cf. settings.COLUMN_CONSTRAINTS)
    date DATE NOT NULL,
    location TEXT NOT NULL,

    min_temp REAL,
    max_temp REAL,
    temp_9am REAL,
    temp_3pm REAL,

    rainfall REAL,
    evaporation REAL,
    sunshine REAL,

    humidity_9am REAL,
    humidity_3pm REAL,

    pressure_9am REAL,
    pressure_3pm REAL,

    wind_gust_dir TEXT,
    wind_gust_speed REAL,
    wind_dir_9am TEXT,
    wind_dir_3pm TEXT,
    wind_speed_9am REAL,
    wind_speed_3pm REAL,

    cloud_9am REAL,
    cloud_3pm REAL,

    rain_today TEXT,
    rain_tomorrow TEXT,

    -- Champs d'audit / colonnes techniques (cf. metadata.TECHNICAL_COLUMNS)
    date_clean TIMESTAMPTZ NOT NULL,
    data_source TEXT NOT NULL,
    run_id TEXT NOT NULL,

    -- Contraintes de valeurs autorisées (cf. settings.COLUMN_CONSTRAINTS)
    CONSTRAINT chk_rain_today CHECK (rain_today IN ('Yes', 'No')),
    CONSTRAINT chk_rain_tomorrow CHECK (rain_tomorrow IN ('Yes', 'No')),
    CONSTRAINT chk_wind_gust_dir CHECK (
        wind_gust_dir IS NULL OR wind_gust_dir IN (
            'N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW'
        )
    ),
    CONSTRAINT chk_wind_dir_9am CHECK (
        wind_dir_9am IS NULL OR wind_dir_9am IN (
            'N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW'
        )
    ),
    CONSTRAINT chk_wind_dir_3pm CHECK (
        wind_dir_3pm IS NULL OR wind_dir_3pm IN (
            'N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW'
        )
    )
);

-- Un import ne doit pas dupliquer une même mesure (date, ville) pour un même run
CREATE UNIQUE INDEX IF NOT EXISTS ux_weather_data_clean_date_location_run
    ON weather_data_clean (date, location, run_id);

-- Utilisé par build_features.load_data_from_postgres pour ne lire que le dernier import
CREATE INDEX IF NOT EXISTS ix_weather_data_clean_date_clean
    ON weather_data_clean (date_clean);
