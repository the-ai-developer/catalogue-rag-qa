-- +migrate Up
-- Catalogue AI — core schema (PostgreSQL 14+)
-- Shared contract for services/api (Go) and services/model-server (Python).
-- NOTE: faiss_id for embeddings == embeddings.id (bigserial, allocated by the API).

CREATE TABLE items (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    sku         text NOT NULL UNIQUE,
    title       text NOT NULL,
    category    text NOT NULL,
    material    text,
    dimensions  jsonb NOT NULL DEFAULT '{}'::jsonb,   -- {"length_cm": 12.0, "width_cm": 6.0, "height_cm": 3.0}
    features    jsonb NOT NULL DEFAULT '[]'::jsonb,   -- ["waterproof", "usb-c"]
    extra       jsonb NOT NULL DEFAULT '{}'::jsonb,   -- free-form spec fields
    status      text NOT NULL DEFAULT 'active'
                CHECK (status IN ('draft', 'active', 'archived')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    deleted_at  timestamptz
);

CREATE TABLE item_assets (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id      uuid NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    kind         text NOT NULL CHECK (kind IN ('image', 'document')),
    storage_path text NOT NULL,
    mime         text NOT NULL,
    width        int,
    height       int,
    bytes        bigint,
    sha256       text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX item_assets_item_idx ON item_assets (item_id);

CREATE TABLE chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id     uuid NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    ordinal     int  NOT NULL,
    source_kind text NOT NULL
                CHECK (source_kind IN ('title', 'spec', 'copy', 'feature', 'dimension')),
    text        text NOT NULL,
    token_count int  NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (item_id, ordinal)
);
CREATE INDEX chunks_item_idx ON chunks (item_id);

-- Embedding metadata. Vectors live in FAISS (model-server); this table is the
-- source of truth for what is indexed and with which model version.
CREATE TABLE embeddings (
    id            bigserial PRIMARY KEY,
    item_id       uuid NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    modality      text NOT NULL CHECK (modality IN ('text', 'image')),
    chunk_id      uuid REFERENCES chunks(id) ON DELETE CASCADE,
    asset_id      uuid REFERENCES item_assets(id) ON DELETE CASCADE,
    model_name    text NOT NULL,
    model_version text NOT NULL,
    dim           int  NOT NULL,
    faiss_id      bigint,                  -- == embeddings.id once indexed
    indexed_at    timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now(),
    CHECK ((modality = 'text' AND chunk_id IS NOT NULL AND asset_id IS NULL) OR
           (modality = 'image' AND asset_id IS NOT NULL AND chunk_id IS NULL))
);
CREATE UNIQUE INDEX embeddings_faiss_id_key ON embeddings (faiss_id) WHERE faiss_id IS NOT NULL;
CREATE INDEX embeddings_item_idx ON embeddings (item_id, modality);

CREATE TABLE ingest_jobs (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id        uuid NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    status         text NOT NULL DEFAULT 'queued'
                   CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    chunks_indexed int  NOT NULL DEFAULT 0,
    images_indexed int  NOT NULL DEFAULT 0,
    error          text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    started_at     timestamptz,
    finished_at    timestamptz
);
CREATE INDEX ingest_jobs_claim_idx ON ingest_jobs (status, created_at) WHERE status = 'queued';

-- ------------------------------ QA (Project 1) ------------------------------

CREATE TABLE qa_queries (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_ref       text NOT NULL,
    question       text NOT NULL,
    scope_item_ids uuid[] NOT NULL DEFAULT '{}',
    use_images     boolean NOT NULL DEFAULT true,
    top_k          int  NOT NULL DEFAULT 6,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE qa_answers (
    id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id               uuid NOT NULL REFERENCES qa_queries(id) ON DELETE CASCADE,
    answer_text            text NOT NULL,
    composition_mode       text NOT NULL CHECK (composition_mode IN ('extractive', 'abstractive')),
    model_version          text NOT NULL,
    citation_check_passed  boolean NOT NULL,
    citation_check_detail  jsonb NOT NULL DEFAULT '{}'::jsonb,
    latency_ms             int  NOT NULL DEFAULT 0,
    created_at             timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX qa_answers_query_idx ON qa_answers (query_id);

CREATE TABLE qa_answer_citations (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    answer_id      uuid NOT NULL REFERENCES qa_answers(id) ON DELETE CASCADE,
    sentence_index int  NOT NULL,
    item_id        uuid NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    chunk_id       uuid REFERENCES chunks(id) ON DELETE SET NULL,
    asset_id       uuid REFERENCES item_assets(id) ON DELETE SET NULL,
    modality       text NOT NULL CHECK (modality IN ('text', 'image')),
    score          real NOT NULL,
    snippet        text NOT NULL
);

-- ------------------------ Descriptions (Project 2) --------------------------

CREATE TABLE spec_sheets (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id     uuid REFERENCES items(id) ON DELETE SET NULL,
    title       text,
    category    text NOT NULL,
    material    text,
    dimensions  jsonb NOT NULL DEFAULT '{}'::jsonb,
    features    jsonb NOT NULL DEFAULT '[]'::jsonb,
    extra       jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_by  text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE generation_jobs (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    spec_id       uuid NOT NULL REFERENCES spec_sheets(id) ON DELETE CASCADE,
    status        text NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'running', 'draft_ready', 'approved',
                                    'rejected', 'published', 'failed')),
    model_name    text,
    model_version text,
    beam_width    int NOT NULL DEFAULT 4,
    max_len       int NOT NULL DEFAULT 192,
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    finished_at   timestamptz
);
CREATE INDEX generation_jobs_claim_idx ON generation_jobs (status, created_at) WHERE status = 'queued';

CREATE TABLE description_drafts (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id     uuid NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
    rank       int  NOT NULL,
    text       text NOT NULL,
    score      real NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id, rank)
);

CREATE TABLE editor_reviews (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      uuid NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
    draft_id    uuid REFERENCES description_drafts(id) ON DELETE SET NULL,
    reviewer    text NOT NULL,
    decision    text NOT NULL CHECK (decision IN ('approve', 'reject', 'edit')),
    edited_text text,
    notes       text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE published_descriptions (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id     uuid NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    job_id      uuid NOT NULL REFERENCES generation_jobs(id) ON DELETE CASCADE,
    draft_id    uuid REFERENCES description_drafts(id) ON DELETE SET NULL,
    text        text NOT NULL,
    approved_by text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX published_descriptions_item_idx ON published_descriptions (item_id, created_at DESC);

-- ------------------------------ Platform ------------------------------------

CREATE TABLE api_keys (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name       text NOT NULL UNIQUE,
    key_hash   text NOT NULL UNIQUE,   -- sha256 hex of the secret
    role       text NOT NULL CHECK (role IN ('viewer', 'editor', 'admin')),
    active     boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE audit_log (
    id        bigserial PRIMARY KEY,
    actor     text NOT NULL,
    action    text NOT NULL,
    entity    text NOT NULL,
    entity_id text NOT NULL,
    detail    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX audit_log_entity_idx ON audit_log (entity, entity_id, created_at DESC);

-- ------------------------------ Triggers ------------------------------------

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER items_updated_at BEFORE UPDATE ON items
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
