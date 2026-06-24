-- prompt_browser schema (PostgreSQL)
-- Run automatically by build_db.py, or manually (PowerShell): psql $env:DATABASE_URL -f schema.sql

CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fast ILIKE '%term%' prompt search

CREATE TABLE IF NOT EXISTS generations (
    id              BIGSERIAL PRIMARY KEY,
    file_path       TEXT UNIQUE NOT NULL,   -- natural key; re-import upserts
    folder          TEXT,
    filename        TEXT,
    model_family    TEXT,
    model_file      TEXT,
    gen_type        TEXT,                   -- t2i / i2i / I2V / T2V
    image_type      TEXT,                   -- SFW / NSFW
    positive_prompt TEXT,
    negative_prompt TEXT,
    width           INTEGER,
    height          INTEGER,
    steps           INTEGER,
    sampler         TEXT,
    scheduler       TEXT,
    guidance        REAL,
    seed            NUMERIC(20,0),          -- ComfyUI seeds can exceed BIGINT
    denoise         REAL,
    lora_count      INTEGER,
    workflow        JSONB,                  -- raw ComfyUI API graph for replay
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gen_model_family ON generations (model_family);
CREATE INDEX IF NOT EXISTS idx_gen_image_type   ON generations (image_type);
CREATE INDEX IF NOT EXISTS idx_gen_gen_type     ON generations (gen_type);
CREATE INDEX IF NOT EXISTS idx_gen_pos_trgm     ON generations USING gin (positive_prompt gin_trgm_ops);

-- Loras normalized so name<->strength stays tied and is queryable.
CREATE TABLE IF NOT EXISTS loras (
    id   SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_loras (
    id            BIGSERIAL PRIMARY KEY,
    generation_id BIGINT  NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    lora_id       INTEGER NOT NULL REFERENCES loras(id) ON DELETE CASCADE,
    strength      REAL
);

CREATE INDEX IF NOT EXISTS idx_genloras_gen  ON generation_loras (generation_id);
CREATE INDEX IF NOT EXISTS idx_genloras_lora ON generation_loras (lora_id);
