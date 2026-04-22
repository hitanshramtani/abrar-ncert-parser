CREATE TABLE IF NOT EXISTS questions (
    id BIGSERIAL PRIMARY KEY,
    question_id TEXT NOT NULL UNIQUE,
    class INTEGER NOT NULL,
    subject TEXT NOT NULL,
    chapter_number INTEGER NOT NULL,
    chapter_name TEXT NOT NULL,
    question_type TEXT,
    question_text TEXT,
    question_latex TEXT,
    answer_text TEXT,
    answer_latex TEXT,
    final_answer TEXT,
    difficulty_rating INTEGER,
    difficulty_reasoning TEXT,
    confidence_score DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS question_parts (
    id BIGSERIAL PRIMARY KEY,
    question_db_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    part_label TEXT,
    part_text TEXT,
    part_latex TEXT,
    answer_text TEXT,
    answer_latex TEXT,
    final_answer TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id BIGSERIAL PRIMARY KEY,
    question_db_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    part_id BIGINT REFERENCES question_parts(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    description TEXT,
    expression TEXT
);

CREATE TABLE IF NOT EXISTS question_images (
    id BIGSERIAL PRIMARY KEY,
    question_db_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    part_id BIGINT REFERENCES question_parts(id) ON DELETE CASCADE,
    image_type TEXT NOT NULL CHECK (image_type IN ('question', 'answer')),
    cloudinary_url TEXT,
    description TEXT,
    latex_extracted TEXT
);

CREATE TABLE IF NOT EXISTS topics_tags (
    id BIGSERIAL PRIMARY KEY,
    question_db_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    UNIQUE (question_db_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_questions_class_subject_chapter
    ON questions (class, subject, chapter_number);

CREATE INDEX IF NOT EXISTS idx_questions_question_type
    ON questions (question_type);

CREATE INDEX IF NOT EXISTS idx_questions_difficulty_rating
    ON questions (difficulty_rating);

CREATE INDEX IF NOT EXISTS idx_questions_confidence_score
    ON questions (confidence_score);
