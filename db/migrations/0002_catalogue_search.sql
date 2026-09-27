-- +migrate Up
-- Catalogue AI — catalogue search and stable pagination.
--
-- Two problems with GET /api/v1/items, both found by running the stack:
--
-- 1. The list endpoint filtered with `$1 <% sku || ' ' || title || ' ' || category`.
--    `<%` is not a PostgreSQL operator, and operator resolution happens at parse
--    time, so *every* list request failed with
--    `ERROR: operator does not exist: text <% text (SQLSTATE 42883)` regardless
--    of the query argument.
--
-- 2. Pagination keyed off `id::text` (unindexable) and then off
--    `(created_at, id)`, which is not unique: several items created in the same
--    transaction or the same millisecond share a `created_at`, so the row
--    comparison silently skipped the ones with a lower uuid. `seq` is a
--    monotonic bigserial and is unique by construction.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Monotonic, unique, gapless-enough keyset for pagination.
ALTER TABLE items ADD COLUMN IF NOT EXISTS seq bigserial;

-- One immutable, indexed concatenation of the searchable text, so every query
-- spells the expression the same way and the trigram index can be used.
ALTER TABLE items
    ADD COLUMN IF NOT EXISTS search_text text
    GENERATED ALWAYS AS (sku || ' ' || title || ' ' || category) STORED;

-- Supports ILIKE '%needle%' and prefix search.
CREATE INDEX IF NOT EXISTS items_search_trgm_idx
    ON items USING gin (search_text gin_trgm_ops);

-- Default listing order, and the exact indexes the three list shapes use.
CREATE INDEX IF NOT EXISTS items_live_seq_idx
    ON items (seq DESC) WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS items_live_status_seq_idx
    ON items (status, seq DESC) WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS items_live_category_seq_idx
    ON items (category, seq DESC) WHERE deleted_at IS NULL;
