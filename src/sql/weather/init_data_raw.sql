CREATE TABLE IF NOT EXISTS weather_data_raw (
    id BIGSERIAL PRIMARY KEY,

    -- Colonnes du dataset brut
    date TEXT,
    location TEXT,
    min_temp TEXT,
    max_temp TEXT,
    rainfall TEXT,
    evaporation TEXT,
    sunshine TEXT,
    wind_gust_dir TEXT,
    wind_gust_speed TEXT,
    wind_dir_9am TEXT,
    wind_dir_3pm TEXT,
    wind_speed_9am TEXT,
    wind_speed_3pm TEXT,
    humidity_9am TEXT,
    humidity_3pm TEXT,
    pressure_9am TEXT,
    pressure_3pm TEXT,
    cloud_9am TEXT,
    cloud_3pm TEXT,
    temp_9am TEXT,
    temp_3pm TEXT,
    rain_today TEXT,
    rain_tomorrow TEXT,

    -- Champs d’audit ajoutés par le task d’import
    date_import TIMESTAMPTZ,
    data_source TEXT,
	run_id TEXT
);

-- 
CREATE INDEX IF NOT EXISTS ix_weather_data_raw_date_import
    ON weather_data_raw (date_import);
