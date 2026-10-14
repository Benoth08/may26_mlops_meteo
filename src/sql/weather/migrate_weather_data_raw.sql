-- Migration additive depuis le schema historique (CamelCase normalise sans
-- underscores) vers le schema courant. Aucune colonne ni ligne n'est supprimee.
ALTER TABLE weather_data_raw
    ADD COLUMN IF NOT EXISTS min_temp TEXT,
    ADD COLUMN IF NOT EXISTS max_temp TEXT,
    ADD COLUMN IF NOT EXISTS wind_gust_dir TEXT,
    ADD COLUMN IF NOT EXISTS wind_gust_speed TEXT,
    ADD COLUMN IF NOT EXISTS wind_dir_9am TEXT,
    ADD COLUMN IF NOT EXISTS wind_dir_3pm TEXT,
    ADD COLUMN IF NOT EXISTS wind_speed_9am TEXT,
    ADD COLUMN IF NOT EXISTS wind_speed_3pm TEXT,
    ADD COLUMN IF NOT EXISTS humidity_9am TEXT,
    ADD COLUMN IF NOT EXISTS humidity_3pm TEXT,
    ADD COLUMN IF NOT EXISTS pressure_9am TEXT,
    ADD COLUMN IF NOT EXISTS pressure_3pm TEXT,
    ADD COLUMN IF NOT EXISTS cloud_9am TEXT,
    ADD COLUMN IF NOT EXISTS cloud_3pm TEXT,
    ADD COLUMN IF NOT EXISTS temp_9am TEXT,
    ADD COLUMN IF NOT EXISTS temp_3pm TEXT,
    ADD COLUMN IF NOT EXISTS rain_today TEXT,
    ADD COLUMN IF NOT EXISTS rain_tomorrow TEXT,
    ADD COLUMN IF NOT EXISTS date_import TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS data_source TEXT,
    ADD COLUMN IF NOT EXISTS run_id TEXT;

UPDATE weather_data_raw
SET
    min_temp = COALESCE(min_temp, mintemp),
    max_temp = COALESCE(max_temp, maxtemp),
    wind_gust_dir = COALESCE(wind_gust_dir, windgustdir),
    wind_gust_speed = COALESCE(wind_gust_speed, windgustspeed),
    wind_dir_9am = COALESCE(wind_dir_9am, winddir9am),
    wind_dir_3pm = COALESCE(wind_dir_3pm, winddir3pm),
    wind_speed_9am = COALESCE(wind_speed_9am, windspeed9am),
    wind_speed_3pm = COALESCE(wind_speed_3pm, windspeed3pm),
    humidity_9am = COALESCE(humidity_9am, humidity9am),
    humidity_3pm = COALESCE(humidity_3pm, humidity3pm),
    pressure_9am = COALESCE(pressure_9am, pressure9am),
    pressure_3pm = COALESCE(pressure_3pm, pressure3pm),
    cloud_9am = COALESCE(cloud_9am, cloud9am),
    cloud_3pm = COALESCE(cloud_3pm, cloud3pm),
    temp_9am = COALESCE(temp_9am, temp9am),
    temp_3pm = COALESCE(temp_3pm, temp3pm),
    rain_today = COALESCE(rain_today, raintoday),
    rain_tomorrow = COALESCE(rain_tomorrow, raintomorrow),
    date_import = COALESCE(date_import, import_date),
    data_source = COALESCE(data_source, source_file),
    run_id = COALESCE(run_id, import_run_id)
WHERE
    min_temp IS NULL OR max_temp IS NULL OR wind_gust_dir IS NULL
    OR wind_gust_speed IS NULL OR wind_dir_9am IS NULL OR wind_dir_3pm IS NULL
    OR wind_speed_9am IS NULL OR wind_speed_3pm IS NULL
    OR humidity_9am IS NULL OR humidity_3pm IS NULL
    OR pressure_9am IS NULL OR pressure_3pm IS NULL
    OR cloud_9am IS NULL OR cloud_3pm IS NULL
    OR temp_9am IS NULL OR temp_3pm IS NULL
    OR rain_today IS NULL OR rain_tomorrow IS NULL
    OR date_import IS NULL OR data_source IS NULL OR run_id IS NULL;

CREATE INDEX IF NOT EXISTS ix_weather_data_raw_date_import
    ON weather_data_raw (date_import);
