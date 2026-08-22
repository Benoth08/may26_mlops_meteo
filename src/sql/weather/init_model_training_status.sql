CREATE TABLE IF NOT EXISTS model_training_status (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    last_training_date TIMESTAMP WITH TIME ZONE,
    last_training_row_count BIGINT DEFAULT 0,
    last_model_version VARCHAR(100),
    last_score DOUBLE PRECISION,
    status VARCHAR(50),
    training_reason VARCHAR(100),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Une seule ligne par modèle : requis pour l'UPSERT
-- ON CONFLICT (model_name) de TrainingStateRepository.update_training_status().
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_training_status_model_name
    ON model_training_status (model_name);
