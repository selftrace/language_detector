
CREATE TABLE model_performance_log (
    run_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accuracy NUMERIC(5, 4) NOT NULL,
    macro_f1 NUMERIC(5, 4) NOT NULL,
    total_samples INT NOT NULL,
    misclassified_count INT NOT NULL,
    best_c NUMERIC(4, 2),
    best_ngram VARCHAR(20),
    best_min_df INT
);

CREATE TABLE predictions_analysis (
    prediction_id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    actual VARCHAR(50) NOT NULL,
    predicted VARCHAR(50) NOT NULL,
    length INT NOT NULL,
    correct BOOLEAN NOT NULL,
    length_group VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_actual_pred ON predictions_analysis (actual, predicted);
CREATE INDEX idx_predictions_correct ON predictions_analysis (correct);
CREATE INDEX idx_predictions_length_group ON predictions_analysis (length_group);