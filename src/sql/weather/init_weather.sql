CREATE TABLE IF NOT EXISTS weather_data_raw (
    id BIGSERIAL PRIMARY KEY,

    -- Colonnes du dataset brut
    date TEXT,
    location TEXT,
    mintemp TEXT,
    maxtemp TEXT,
    rainfall TEXT,
    evaporation TEXT,
    sunshine TEXT,
    windgustdir TEXT,
    windgustspeed TEXT,
    winddir9am TEXT,
    winddir3pm TEXT,
    windspeed9am TEXT,
    windspeed3pm TEXT,
    humidity9am TEXT,
    humidity3pm TEXT,
    pressure9am TEXT,
    pressure3pm TEXT,
    cloud9am TEXT,
    cloud3pm TEXT,
    temp9am TEXT,
    temp3pm TEXT,
    raintoday TEXT,
    raintomorrow TEXT,

    -- Champs d’audit ajoutés par le task d’import
    import_date TIMESTAMPTZ,
    import_run_id TEXT,
    source_file TEXT
);
