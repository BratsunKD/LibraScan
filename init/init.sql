CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP NOT NULL,
    img_url TEXT NOT NULL,
    model_answer INTEGER NOT NULL
);

CREATE TABLE user_answers (
    id SERIAL PRIMARY KEY,
    prediction_hash TEXT,
    user_answer INTEGER NOT NULL,
    answered_at TIMESTAMP NOT NULL
);
