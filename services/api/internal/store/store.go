// Package store is the PostgreSQL data layer (schema: db/migrations).
package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"

	"catalogue-ai/services/api/internal/apperr"
	"catalogue-ai/services/api/internal/workflow"
)

type Store struct{ Pool *pgxpool.Pool }

func New(pool *pgxpool.Pool) *Store { return &Store{Pool: pool} }

var ErrNotFound = errors.New("not found")

// ------------------------------- domain types -------------------------------

type Counts struct {
	Chunks          int `json:"chunks"`
	EmbeddingsText  int `json:"embeddings_text"`
	EmbeddingsImage int `json:"embeddings_image"`
}

type Asset struct {
	ID        string    `json:"id"`
	Kind      string    `json:"kind"`
	URL       string    `json:"url"`
	Mime      string    `json:"mime"`
	Width     *int      `json:"width"`
	Height    *int      `json:"height"`
	Bytes     int64     `json:"bytes"`
	SHA256    string    `json:"sha256"`
	CreatedAt time.Time `json:"created_at"`
}

type Item struct {
	ID         string         `json:"id"`
	SKU        string         `json:"sku"`
	Title      string         `json:"title"`
	Category   string         `json:"category"`
	Material   *string        `json:"material"`
	Dimensions map[string]any `json:"dimensions"`
	Features   []string       `json:"features"`
	Extra      map[string]any `json:"extra"`
	Status     string         `json:"status"`
	CreatedAt  time.Time      `json:"created_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
	Assets     []Asset        `json:"assets"`
	Counts     Counts         `json:"counts"`
}

type IngestJob struct {
	ID            string     `json:"id"`
	ItemID        string     `json:"item_id"`
	Status        string     `json:"status"`
	ChunksIndexed int        `json:"chunks_indexed"`
	ImagesIndexed int        `json:"images_indexed"`
	Error         *string    `json:"error"`
	CreatedAt     time.Time  `json:"created_at"`
	StartedAt     *time.Time `json:"started_at"`
	FinishedAt    *time.Time `json:"finished_at"`
}

type ChunkRow struct {
	ID      string
	Text    string
	Ordinal int
}

type Query struct {
	ID           string    `json:"id"`
	UserRef      string    `json:"user_ref"`
	Question     string    `json:"question"`
	ScopeItemIDs []string  `json:"scope_item_ids"`
	UseImages    bool      `json:"use_images"`
	TopK         int       `json:"top_k"`
	CreatedAt    time.Time `json:"created_at"`
}

type CitationRow struct {
	SentenceIndex int     `json:"sentence_index"`
	ItemID        string  `json:"item_id"`
	SKU           string  `json:"sku"`
	ChunkID       *string `json:"chunk_id"`
	AssetID       *string `json:"asset_id"`
	Modality      string  `json:"modality"`
	Score         float64 `json:"score"`
	Snippet       string  `json:"snippet"`
}

// Answer is a stored QA answer plus the question it answers. Question and
// UserRef are joined in from qa_queries: the UI renders both in its history
// table, and omitting them produced a permanently blank column.
type Answer struct {
	ID            string         `json:"answer_id"`
	QueryID       string         `json:"query_id"`
	Question      string         `json:"question"`
	UserRef       string         `json:"user_ref"`
	Answer        string         `json:"answer"`
	Composition   string         `json:"composition"`
	ModelVersion  string         `json:"model_version"`
	CitationCheck map[string]any `json:"citation_check"`
	// CitationCheckPass used to be tagged `json:"-"`, so the API never emitted
	// citation_check_passed and every history row rendered as "withheld".
	CitationCheckPass bool          `json:"citation_check_passed"`
	LatencyMS         int           `json:"latency_ms"`
	CreatedAt         time.Time     `json:"created_at"`
	Citations         []CitationRow `json:"citations"`
}

type SpecSheet struct {
	ID         string         `json:"id"`
	ItemID     *string        `json:"item_id"`
	Title      *string        `json:"title"`
	Category   string         `json:"category"`
	Material   *string        `json:"material"`
	Dimensions map[string]any `json:"dimensions"`
	Features   []string       `json:"features"`
	Extra      map[string]any `json:"extra"`
	CreatedBy  string         `json:"created_by"`
	CreatedAt  time.Time      `json:"created_at"`
}

type Draft struct {
	ID    string  `json:"id"`
	JobID string  `json:"job_id"`
	Rank  int     `json:"rank"`
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

type Review struct {
	ID         string    `json:"id"`
	JobID      string    `json:"job_id"`
	DraftID    *string   `json:"draft_id"`
	Reviewer   string    `json:"reviewer"`
	Decision   string    `json:"decision"`
	EditedText *string   `json:"edited_text"`
	Notes      *string   `json:"notes"`
	CreatedAt  time.Time `json:"created_at"`
}

type Published struct {
	ID         string    `json:"id"`
	ItemID     string    `json:"item_id"`
	JobID      string    `json:"job_id"`
	DraftID    *string   `json:"draft_id,omitempty"`
	Text       string    `json:"text"`
	ApprovedBy string    `json:"approved_by"`
	CreatedAt  time.Time `json:"created_at"`
}

type GenerationJob struct {
	ID           string      `json:"id"`
	SpecID       string      `json:"spec_id"`
	Status       string      `json:"status"`
	ModelName    *string     `json:"model_name"`
	ModelVersion *string     `json:"model_version"`
	BeamWidth    int         `json:"beam_width"`
	MaxLen       int         `json:"max_len"`
	Error        *string     `json:"error"`
	CreatedAt    time.Time   `json:"created_at"`
	StartedAt    *time.Time  `json:"started_at"`
	FinishedAt   *time.Time  `json:"finished_at"`
	Spec         *SpecSheet  `json:"spec,omitempty"`
	Drafts       []Draft     `json:"drafts"`
	Reviews      []Review    `json:"reviews"`
	Published    []Published `json:"published"`
}

// ------------------------------- helpers ------------------------------------

// marshalJSON encodes a value for a jsonb column, never emitting JSON null.
//
// A nil []string or map[string]any boxed into `any` is a *typed* nil, so the
// `v == nil` guard misses it and json.Marshal writes `null`. That bypasses the
// column's NOT NULL DEFAULT '[]', and the read path then hands the API a nil
// slice, which the UI calls .join() on. Empty containers must serialise as [] / {}.
func marshalJSON(v any) []byte {
	if v == nil {
		return []byte("{}")
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Slice, reflect.Array, reflect.Map:
		if rv.IsNil() {
			if rv.Kind() == reflect.Slice {
				return []byte("[]")
			}
			return []byte("{}")
		}
	}
	b, err := json.Marshal(v)
	if err != nil || string(b) == "null" {
		return []byte("{}")
	}
	return b
}

// marshal keeps the original name for callers that only ever pass maps.
func marshal(v any) []byte { return marshalJSON(v) }

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// ------------------------------- items --------------------------------------

// ErrDuplicateSKU is returned when an insert violates the items.sku unique
// index. Detected from the SQLSTATE rather than by substring-matching the
// driver error text.
var ErrDuplicateSKU = errors.New("sku already exists")

// sqlStateUniqueViolation is the PostgreSQL SQLSTATE for a unique-constraint
// violation. Matched against the typed *pgconn.PgError rather than by
// substring-matching the driver message, which breaks when the message or
// locale changes.
const sqlStateUniqueViolation = "23505"

func isUniqueViolation(err error) bool {
	var pgErr *pgconn.PgError
	return errors.As(err, &pgErr) && pgErr.Code == sqlStateUniqueViolation
}

func (s *Store) CreateItem(ctx context.Context, it Item) (Item, error) {
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO items (sku, title, category, material, dimensions, features, extra, status)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		RETURNING id, created_at, updated_at`,
		it.SKU, it.Title, it.Category, it.Material, marshalJSON(it.Dimensions),
		marshalJSON(it.Features), marshalJSON(it.Extra), it.Status).
		Scan(&it.ID, &it.CreatedAt, &it.UpdatedAt)
	if err != nil {
		if isUniqueViolation(err) {
			return it, fmt.Errorf("%w: %v", ErrDuplicateSKU, err)
		}
		return it, err
	}
	return s.GetItem(ctx, it.ID)
}

func (s *Store) GetItem(ctx context.Context, id string) (Item, error) {
	var it Item
	var dims, feats, extra []byte
	err := s.Pool.QueryRow(ctx, `
		SELECT id, sku, title, category, material, dimensions::text, features::text,
		       extra::text, status, created_at, updated_at
		FROM items WHERE id=$1 AND deleted_at IS NULL`, id).
		Scan(&it.ID, &it.SKU, &it.Title, &it.Category, &it.Material, &dims,
			&feats, &extra, &it.Status, &it.CreatedAt, &it.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return it, ErrNotFound
	}
	if err != nil {
		return it, err
	}
	it.Dimensions = map[string]any{}
	it.Features = []string{}
	it.Extra = map[string]any{}
	_ = json.Unmarshal(dims, &it.Dimensions)
	_ = json.Unmarshal(feats, &it.Features)
	_ = json.Unmarshal(extra, &it.Extra)
	it.Assets, _ = s.ListAssets(ctx, id)
	it.Counts, _ = s.ItemCounts(ctx, id)
	return it, nil
}

type ListFilter struct {
	Query    string
	Category string
	Status   string
	Limit    int
	// Cursor is the opaque value handed back as `next_cursor`.
	Cursor string
	// CursorSeq is the decoded cursor: the `items.seq` of the last row on the
	// previous page. Passed as NULL when absent, because casting an empty
	// string to bigint raises and the predicate is evaluated even on page one.
	CursorSeq *int64
}

// ListItemsPage is the next-page cursor for ListItems. The zero value means
// "no more pages".
type ListItemsPage struct {
	Seq int64
}

// Cursor renders the page as the opaque `next_cursor` string.
func (p ListItemsPage) Cursor() string {
	if p.Seq <= 0 {
		return ""
	}
	return strconv.FormatInt(p.Seq, 10)
}

// ParseListItemsPage decodes an opaque cursor. An empty or malformed cursor
// means "start from the beginning" rather than an error: a stale bookmark should
// not turn into a 500.
func ParseListItemsPage(cursor string) int64 {
	if cursor == "" {
		return 0
	}
	n, err := strconv.ParseInt(cursor, 10, 64)
	if err != nil || n < 0 {
		return 0
	}
	return n
}

func (s *Store) ListItems(ctx context.Context, f ListFilter) ([]Item, ListItemsPage, error) {
	if f.Limit <= 0 || f.Limit > 100 {
		f.Limit = 20
	}
	// The caller's text is matched literally, so a query containing % or _ is a
	// search for those characters and not a wildcard pattern.
	pattern := "%" + escapeLike(f.Query) + "%"
	rows, err := s.Pool.Query(ctx, `
		SELECT id, seq, sku, title, category, material, dimensions::text,
		       features::text, extra::text, status, created_at, updated_at
		FROM items
		WHERE deleted_at IS NULL
		  AND ($1::text = '' OR search_text ILIKE $2 ESCAPE '\')
		  AND ($3::text = '' OR category = $3)
		  AND ($4::text = '' OR status = $4)
		  AND ($5::bigint IS NULL OR seq < $5)
		ORDER BY seq DESC
		LIMIT $6`,
		f.Query, pattern, f.Category, f.Status, f.CursorSeq, f.Limit+1)
	if err != nil {
		return nil, ListItemsPage{}, err
	}
	defer rows.Close()
	items := []Item{}
	seqs := make([]int64, 0, f.Limit+1)
	for rows.Next() {
		var it Item
		var seq int64
		var dims, feats, extra []byte
		if err := rows.Scan(&it.ID, &seq, &it.SKU, &it.Title, &it.Category, &it.Material,
			&dims, &feats, &extra, &it.Status, &it.CreatedAt, &it.UpdatedAt); err != nil {
			return nil, ListItemsPage{}, err
		}
		it.Dimensions = map[string]any{}
		it.Features = []string{}
		it.Extra = map[string]any{}
		_ = json.Unmarshal(dims, &it.Dimensions)
		_ = json.Unmarshal(feats, &it.Features)
		_ = json.Unmarshal(extra, &it.Extra)
		items = append(items, it)
		seqs = append(seqs, seq)
	}
	if err := rows.Err(); err != nil {
		return nil, ListItemsPage{}, err
	}
	next := ListItemsPage{}
	if len(items) > f.Limit {
		items = items[:f.Limit]
		seqs = seqs[:f.Limit]
		next = ListItemsPage{Seq: seqs[len(seqs)-1]}
	}
	// One query each for the whole page, not two per item: at limit=20 the
	// per-item version issued 61 statements for one list request.
	ids := make([]string, 0, len(items))
	for i := range items {
		ids = append(ids, items[i].ID)
	}
	assets, err := s.assetsForItems(ctx, ids)
	if err != nil {
		return nil, ListItemsPage{}, err
	}
	counts, err := s.countsForItems(ctx, ids)
	if err != nil {
		return nil, ListItemsPage{}, err
	}
	for i := range items {
		items[i].Assets = assets[items[i].ID]
		if items[i].Assets == nil {
			items[i].Assets = []Asset{}
		}
		items[i].Counts = counts[items[i].ID]
	}
	return items, next, nil
}

// escapeLike neutralises LIKE/ILIKE metacharacters so a user-supplied query is
// matched literally.
func escapeLike(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`)
	return r.Replace(s)
}

// assetsForItems loads every asset for a page of items in one statement.
func (s *Store) assetsForItems(ctx context.Context, ids []string) (map[string][]Asset, error) {
	out := map[string][]Asset{}
	if len(ids) == 0 {
		return out, nil
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT item_id, id, kind, storage_path, mime, width, height, bytes, sha256, created_at
		FROM item_assets WHERE item_id = ANY($1::uuid[]) ORDER BY item_id, created_at`, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var itemID string
		var a Asset
		if err := rows.Scan(&itemID, &a.ID, &a.Kind, &a.URL, &a.Mime, &a.Width,
			&a.Height, &a.Bytes, &a.SHA256, &a.CreatedAt); err != nil {
			return nil, err
		}
		out[itemID] = append(out[itemID], a)
	}
	return out, rows.Err()
}

// countsForItems aggregates chunk/embedding counts for a page in one statement.
//
// The `$1::uuid[]` casts are load-bearing: pgx encodes a Go []string as
// text[], and `uuid = ANY(text[])` is a type error, not a coercion. Without the
// cast every list request 500s.
func (s *Store) countsForItems(ctx context.Context, ids []string) (map[string]Counts, error) {
	out := map[string]Counts{}
	if len(ids) == 0 {
		return out, nil
	}
	rows, err := s.Pool.Query(ctx, `
		WITH c AS (
		  SELECT item_id, count(*) AS chunks FROM chunks
		  WHERE item_id = ANY($1::uuid[]) GROUP BY item_id
		), e AS (
		  SELECT item_id,
		         count(*) FILTER (WHERE modality='text')  AS t,
		         count(*) FILTER (WHERE modality='image') AS i
		  FROM embeddings WHERE item_id = ANY($1::uuid[]) GROUP BY item_id
		)
		SELECT i.id,
		       coalesce(c.chunks,0), coalesce(e.t,0), coalesce(e.i,0)
		FROM unnest($1::uuid[]) AS i(id)
		LEFT JOIN c ON c.item_id = i.id
		LEFT JOIN e ON e.item_id = i.id`, ids)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var id string
		var c Counts
		if err := rows.Scan(&id, &c.Chunks, &c.EmbeddingsText, &c.EmbeddingsImage); err != nil {
			return nil, err
		}
		out[id] = c
	}
	return out, rows.Err()
}

type ItemPatch struct {
	Title      *string
	Category   *string
	Material   *string
	Dimensions map[string]any
	Features   []string
	Extra      map[string]any
	Status     *string
}

func (s *Store) UpdateItem(ctx context.Context, id string, p ItemPatch) (Item, error) {
	_, err := s.Pool.Exec(ctx, `
		UPDATE items SET
		  title      = COALESCE($2, title),
		  category   = COALESCE($3, category),
		  material   = COALESCE($4, material),
		  dimensions = COALESCE($5::jsonb, dimensions),
		  features   = COALESCE($6::jsonb, features),
		  extra      = COALESCE($7::jsonb, extra),
		  status     = COALESCE($8, status)
		WHERE id=$1 AND deleted_at IS NULL`,
		id, p.Title, p.Category, p.Material,
		optionalJSON(p.Dimensions), optionalJSON(p.Features),
		optionalJSON(p.Extra), p.Status)
	if err != nil {
		return Item{}, err
	}
	return s.GetItem(ctx, id)
}

// optionalJSON returns SQL NULL when the field was absent from the PATCH, so
// COALESCE keeps the stored value. A present-but-empty value is encoded as
// []/{} and overwrites, which is what "clear this field" means.
func optionalJSON(v any) []byte {
	if v == nil {
		return nil
	}
	return marshalJSON(v)
}

func (s *Store) SoftDeleteItem(ctx context.Context, id string) error {
	tag, err := s.Pool.Exec(ctx, `
		UPDATE items SET deleted_at=now(), status='archived'
		WHERE id=$1 AND deleted_at IS NULL`, id)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	return nil
}

func (s *Store) AddAsset(ctx context.Context, itemID string, a Asset) (Asset, error) {
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO item_assets (item_id, kind, storage_path, mime, width, height, bytes, sha256)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
		RETURNING id, created_at`,
		itemID, a.Kind, a.URL, a.Mime, a.Width, a.Height, a.Bytes, a.SHA256).
		Scan(&a.ID, &a.CreatedAt)
	return a, err
}

func (s *Store) ListAssets(ctx context.Context, itemID string) ([]Asset, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, kind, storage_path, mime, width, height, bytes, sha256, created_at
		FROM item_assets WHERE item_id=$1 ORDER BY created_at`, itemID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Asset{}
	for rows.Next() {
		var a Asset
		if err := rows.Scan(&a.ID, &a.Kind, &a.URL, &a.Mime, &a.Width, &a.Height,
			&a.Bytes, &a.SHA256, &a.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, nil
}

func (s *Store) ItemCounts(ctx context.Context, itemID string) (Counts, error) {
	var c Counts
	err := s.Pool.QueryRow(ctx, `
		SELECT
		  (SELECT count(*) FROM chunks WHERE item_id=$1),
		  (SELECT count(*) FROM embeddings WHERE item_id=$1 AND modality='text'),
		  (SELECT count(*) FROM embeddings WHERE item_id=$1 AND modality='image')`,
		itemID).Scan(&c.Chunks, &c.EmbeddingsText, &c.EmbeddingsImage)
	return c, err
}

// ------------------------------- ingest pipeline ----------------------------

func (s *Store) CreateIngestJob(ctx context.Context, itemID string) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx,
		`INSERT INTO ingest_jobs (item_id) VALUES ($1) RETURNING id`, itemID).Scan(&id)
	return id, err
}

func (s *Store) GetIngestJob(ctx context.Context, id string) (IngestJob, error) {
	var j IngestJob
	err := s.Pool.QueryRow(ctx, `
		SELECT id, item_id, status, chunks_indexed, images_indexed, error,
		       created_at, started_at, finished_at
		FROM ingest_jobs WHERE id=$1`, id).
		Scan(&j.ID, &j.ItemID, &j.Status, &j.ChunksIndexed, &j.ImagesIndexed,
			&j.Error, &j.CreatedAt, &j.StartedAt, &j.FinishedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	return j, err
}

// ClaimIngestJob atomically claims the oldest queued job (safe with N replicas).
func (s *Store) ClaimIngestJob(ctx context.Context) (IngestJob, error) {
	var j IngestJob
	err := s.Pool.QueryRow(ctx, `
		UPDATE ingest_jobs SET status='running', started_at=now()
		WHERE id = (SELECT id FROM ingest_jobs WHERE status='queued'
		            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
		RETURNING id, item_id, status, chunks_indexed, images_indexed, error,
		          created_at, started_at, finished_at`).
		Scan(&j.ID, &j.ItemID, &j.Status, &j.ChunksIndexed, &j.ImagesIndexed,
			&j.Error, &j.CreatedAt, &j.StartedAt, &j.FinishedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	return j, err
}

// FinishIngestJob settles a claimed job. The `status='running'` predicate is the
// transition guard: a second worker, a retry, or a late finish cannot rewrite a
// job that already reached a terminal state.
func (s *Store) FinishIngestJob(ctx context.Context, id, status string, chunks,
	images int, errMsg *string) error {
	_, err := s.Pool.Exec(ctx, `
		UPDATE ingest_jobs SET status=$2, chunks_indexed=$3, images_indexed=$4,
		       error=$5, finished_at=now()
		WHERE id=$1 AND status='running'`, id, status, chunks, images, errMsg)
	return err
}

// ReplaceChunks swaps the item's chunks in one transaction and returns them.
func (s *Store) ReplaceChunks(ctx context.Context, itemID string, texts []string,
	sources []string, tokenCounts []int) ([]ChunkRow, error) {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if _, err := tx.Exec(ctx, `DELETE FROM chunks WHERE item_id=$1`, itemID); err != nil {
		return nil, err
	}
	out := make([]ChunkRow, 0, len(texts))
	for i := range texts {
		var row ChunkRow
		if err := tx.QueryRow(ctx, `
			INSERT INTO chunks (item_id, ordinal, source_kind, text, token_count)
			VALUES ($1,$2,$3,$4,$5) RETURNING id`,
			itemID, i, sources[i], texts[i], tokenCounts[i]).
			Scan(&row.ID); err != nil {
			return nil, err
		}
		row.Text, row.Ordinal = texts[i], i
		out = append(out, row)
	}
	return out, tx.Commit(ctx)
}

// ClearEmbeddings removes an item's embedding rows and returns old faiss ids.
func (s *Store) ClearEmbeddings(ctx context.Context, itemID string) ([]int64, error) {
	rows, err := s.Pool.Query(ctx,
		`DELETE FROM embeddings WHERE item_id=$1 RETURNING coalesce(faiss_id, id)`, itemID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// InsertEmbedding allocates embeddings.id (= faiss_id) for one vector.
func (s *Store) InsertEmbedding(ctx context.Context, itemID, modality string,
	chunkID, assetID *string, model, version string, dim int) (int64, error) {
	var id int64
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO embeddings (item_id, modality, chunk_id, asset_id,
		                        model_name, model_version, dim)
		VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id`,
		itemID, modality, chunkID, assetID, model, version, dim).Scan(&id)
	return id, err
}

func (s *Store) MarkIndexed(ctx context.Context, faissIDs []int64) error {
	if len(faissIDs) == 0 {
		return nil
	}
	_, err := s.Pool.Exec(ctx,
		`UPDATE embeddings SET indexed_at=now() WHERE id = ANY($1)`, faissIDs)
	return err
}

// ------------------------------- QA -----------------------------------------

func (s *Store) InsertQuery(ctx context.Context, q Query) (string, error) {
	var id string
	// Hand pgx the Go slice: it encodes the uuid[] literal itself. Building the
	// literal by concatenation corrupts any id containing a comma or brace and
	// is not parameterised.
	scope := q.ScopeItemIDs
	if scope == nil {
		scope = []string{}
	}
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO qa_queries (user_ref, question, scope_item_ids, use_images, top_k)
		VALUES ($1,$2,$3,$4,$5) RETURNING id`,
		q.UserRef, q.Question, scope, q.UseImages, q.TopK).Scan(&id)
	return id, err
}

func (s *Store) InsertAnswer(ctx context.Context, a Answer) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO qa_answers (query_id, answer_text, composition_mode, model_version,
		                        citation_check_passed, citation_check_detail, latency_ms)
		VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id`,
		a.QueryID, a.Answer, a.Composition, a.ModelVersion,
		a.CitationCheckPass, marshal(a.CitationCheck), a.LatencyMS).Scan(&id)
	return id, err
}

func (s *Store) InsertCitation(ctx context.Context, answerID string, c CitationRow) error {
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO qa_answer_citations
		  (answer_id, sentence_index, item_id, chunk_id, asset_id, modality, score, snippet)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
		answerID, c.SentenceIndex, c.ItemID, c.ChunkID, c.AssetID,
		c.Modality, c.Score, c.Snippet)
	return err
}

func (s *Store) GetAnswer(ctx context.Context, id string) (Answer, error) {
	var a Answer
	var detail []byte
	err := s.Pool.QueryRow(ctx, `
		SELECT a.id, a.query_id, q.question, q.user_ref, a.answer_text, a.composition_mode,
		       a.model_version, a.citation_check_passed, a.citation_check_detail::text,
		       a.latency_ms, a.created_at
		FROM qa_answers a JOIN qa_queries q ON q.id = a.query_id
		WHERE a.id=$1`, id).
		Scan(&a.ID, &a.QueryID, &a.Question, &a.UserRef, &a.Answer, &a.Composition,
			&a.ModelVersion, &a.CitationCheckPass, &detail, &a.LatencyMS, &a.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return a, ErrNotFound
	}
	if err != nil {
		return a, err
	}
	_ = json.Unmarshal(detail, &a.CitationCheck)
	rows, err := s.Pool.Query(ctx, `
		SELECT c.sentence_index, c.item_id, i.sku, c.chunk_id, c.asset_id,
		       c.modality, c.score, c.snippet
		FROM qa_answer_citations c JOIN items i ON i.id=c.item_id
		WHERE c.answer_id=$1 ORDER BY c.sentence_index`, id)
	if err != nil {
		return a, err
	}
	defer rows.Close()
	a.Citations = []CitationRow{}
	for rows.Next() {
		var c CitationRow
		if err := rows.Scan(&c.SentenceIndex, &c.ItemID, &c.SKU, &c.ChunkID,
			&c.AssetID, &c.Modality, &c.Score, &c.Snippet); err != nil {
			return a, err
		}
		a.Citations = append(a.Citations, c)
	}
	return a, rows.Err()
}

func (s *Store) ListAnswers(ctx context.Context, userRef string, limit int) ([]Answer, int, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	rows, err := s.Pool.Query(ctx, `
		SELECT a.id, a.query_id, q.question, q.user_ref, a.answer_text, a.composition_mode,
		       a.model_version, a.citation_check_passed, a.citation_check_detail::text,
		       a.latency_ms, a.created_at
		FROM qa_answers a JOIN qa_queries q ON q.id = a.query_id
		WHERE ($1='' OR q.user_ref=$1)
		ORDER BY a.created_at DESC, a.id DESC
		LIMIT $2`, userRef, limit+1)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	out := []Answer{}
	for rows.Next() {
		var a Answer
		var detail []byte
		if err := rows.Scan(&a.ID, &a.QueryID, &a.Question, &a.UserRef, &a.Answer,
			&a.Composition, &a.ModelVersion, &a.CitationCheckPass, &detail,
			&a.LatencyMS, &a.CreatedAt); err != nil {
			return nil, 0, err
		}
		_ = json.Unmarshal(detail, &a.CitationCheck)
		a.Citations = []CitationRow{}
		out = append(out, a)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, err
	}
	next := 0
	if len(out) > limit {
		out = out[:limit]
		next = limit
	}
	return out, next, nil
}

// ------------------------------- descriptions -------------------------------

func (s *Store) CreateSpec(ctx context.Context, sp SpecSheet) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO spec_sheets (item_id, title, category, material, dimensions,
		                         features, extra, created_by)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING id`,
		sp.ItemID, sp.Title, sp.Category, sp.Material, marshalJSON(sp.Dimensions),
		marshalJSON(sp.Features), marshalJSON(sp.Extra), sp.CreatedBy).Scan(&id)
	return id, err
}

func (s *Store) CreateGenerationJob(ctx context.Context, specID string, beam,
	maxLen int) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO generation_jobs (spec_id, beam_width, max_len) VALUES ($1,$2,$3)
		RETURNING id`, specID, beam, maxLen).Scan(&id)
	return id, err
}

func (s *Store) ClaimGenerationJob(ctx context.Context) (GenerationJob, error) {
	var j GenerationJob
	err := s.Pool.QueryRow(ctx, `
		UPDATE generation_jobs SET status='running', started_at=now()
		WHERE id = (SELECT id FROM generation_jobs WHERE status='queued'
		            ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
		RETURNING id, spec_id, status, beam_width, max_len, created_at, started_at`).
		Scan(&j.ID, &j.SpecID, &j.Status, &j.BeamWidth, &j.MaxLen,
			&j.CreatedAt, &j.StartedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	return j, err
}

func (s *Store) FinishGenerationJob(ctx context.Context, id, status string,
	model, version *string, errMsg *string) error {
	_, err := s.Pool.Exec(ctx, `
		UPDATE generation_jobs SET status=$2, model_name=$3, model_version=$4,
		       error=$5, finished_at=now()
		WHERE id=$1 AND status='running'`, id, status, model, version, errMsg)
	return err
}

func (s *Store) InsertDrafts(ctx context.Context, jobID string, drafts []Draft) error {
	for _, d := range drafts {
		if _, err := s.Pool.Exec(ctx, `
			INSERT INTO description_drafts (job_id, rank, text, score)
			VALUES ($1,$2,$3,$4) ON CONFLICT (job_id, rank) DO UPDATE
			SET text=EXCLUDED.text, score=EXCLUDED.score`,
			jobID, d.Rank, d.Text, d.Score); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) GetSpec(ctx context.Context, id string) (SpecSheet, error) {
	var sp SpecSheet
	var dims, feats, extra []byte
	err := s.Pool.QueryRow(ctx, `
		SELECT id, item_id, title, category, material, dimensions::text,
		       features::text, extra::text, created_by, created_at
		FROM spec_sheets WHERE id=$1`, id).
		Scan(&sp.ID, &sp.ItemID, &sp.Title, &sp.Category, &sp.Material, &dims,
			&feats, &extra, &sp.CreatedBy, &sp.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return sp, ErrNotFound
	}
	sp.Dimensions = map[string]any{}
	sp.Features = []string{}
	sp.Extra = map[string]any{}
	_ = json.Unmarshal(dims, &sp.Dimensions)
	_ = json.Unmarshal(feats, &sp.Features)
	_ = json.Unmarshal(extra, &sp.Extra)
	return sp, err
}

func (s *Store) GetGenerationJob(ctx context.Context, id string) (GenerationJob, error) {
	var j GenerationJob
	err := s.Pool.QueryRow(ctx, `
		SELECT id, spec_id, status, model_name, model_version, beam_width, max_len,
		       error, created_at, started_at, finished_at
		FROM generation_jobs WHERE id=$1`, id).
		Scan(&j.ID, &j.SpecID, &j.Status, &j.ModelName, &j.ModelVersion,
			&j.BeamWidth, &j.MaxLen, &j.Error, &j.CreatedAt, &j.StartedAt,
			&j.FinishedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return j, ErrNotFound
	}
	if err != nil {
		return j, err
	}
	j.Spec, _ = s.getSpecOrNil(ctx, j.SpecID)
	j.Drafts, _ = s.ListDrafts(ctx, id)
	j.Reviews, _ = s.ListReviews(ctx, id)
	j.Published, _ = s.ListPublishedByJob(ctx, id)
	return j, nil
}

func (s *Store) getSpecOrNil(ctx context.Context, id string) (*SpecSheet, error) {
	sp, err := s.GetSpec(ctx, id)
	if err != nil {
		return nil, err
	}
	return &sp, nil
}

func (s *Store) ListGenerationJobs(ctx context.Context, status string, limit int) ([]GenerationJob, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	// One joined statement instead of a GetGenerationJob fan-out (which cost
	// 1 + 4 queries per row: ~101 statements for a page of 20).
	rows, err := s.Pool.Query(ctx, `
		SELECT g.id, g.spec_id, g.status, g.model_name, g.model_version, g.beam_width,
		       g.max_len, g.error, g.created_at, g.started_at, g.finished_at,
		       sp.item_id, sp.title, sp.category, sp.material, sp.dimensions::text,
		       sp.features::text, sp.extra::text, sp.created_by, sp.created_at,
		       d.id, d.rank, d.text, d.score,
		       r.id, r.draft_id, r.reviewer, r.decision, r.edited_text, r.notes, r.created_at,
		       p.id, p.item_id, p.job_id, p.draft_id, p.text, p.approved_by, p.created_at
		FROM generation_jobs g
		LEFT JOIN spec_sheets sp ON sp.id = g.spec_id
		LEFT JOIN description_drafts d ON d.job_id = g.id
		LEFT JOIN editor_reviews r ON r.job_id = g.id
		LEFT JOIN published_descriptions p ON p.job_id = g.id
		WHERE ($1='' OR g.status=$1)
		ORDER BY g.created_at DESC
		LIMIT $2`, status, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	order := []string{}
	byID := map[string]*GenerationJob{}
	for rows.Next() {
		var j GenerationJob
		var spItemID, spTitle, spCategory, spMaterial *string
		var spDims, spFeats, spExtra []byte
		var spCreatedBy string
		var spCreatedAt time.Time
		var dID *string
		var dRank *int
		var dText *string
		var dScore *float64
		var rID, rDraftID, rReviewer, rDecision, rEdited, rNotes *string
		var rCreatedAt *time.Time
		var pID, pItemID, pJobID, pDraftID, pText, pApprovedBy *string
		var pCreatedAt *time.Time
		if err := rows.Scan(&j.ID, &j.SpecID, &j.Status, &j.ModelName, &j.ModelVersion,
			&j.BeamWidth, &j.MaxLen, &j.Error, &j.CreatedAt, &j.StartedAt, &j.FinishedAt,
			&spItemID, &spTitle, &spCategory, &spMaterial, &spDims, &spFeats, &spExtra,
			&spCreatedBy, &spCreatedAt,
			&dID, &dRank, &dText, &dScore,
			&rID, &rDraftID, &rReviewer, &rDecision, &rEdited, &rNotes, &rCreatedAt,
			&pID, &pItemID, &pJobID, &pDraftID, &pText, &pApprovedBy, &pCreatedAt,
		); err != nil {
			return nil, err
		}
		cur, seen := byID[j.ID]
		if !seen {
			j.Drafts = []Draft{}
			j.Reviews = []Review{}
			j.Published = []Published{}
			if spCreatedBy != "" {
				sp := SpecSheet{ID: j.SpecID, ItemID: spItemID, Title: spTitle,
					Category: derefOr(spCategory, ""), Material: spMaterial,
					CreatedBy: spCreatedBy, CreatedAt: spCreatedAt}
				sp.Dimensions = map[string]any{}
				sp.Features = []string{}
				sp.Extra = map[string]any{}
				_ = json.Unmarshal(spDims, &sp.Dimensions)
				_ = json.Unmarshal(spFeats, &sp.Features)
				_ = json.Unmarshal(spExtra, &sp.Extra)
				j.Spec = &sp
			}
			cur = &j
			byID[j.ID] = cur
			order = append(order, j.ID)
		}
		if dID != nil {
			cur.Drafts = append(cur.Drafts, Draft{ID: *dID, JobID: j.ID,
				Rank: derefInt(dRank), Text: derefStr(dText),
				Score: derefFloat(dScore)})
		}
		if rID != nil {
			cur.Reviews = append(cur.Reviews, Review{ID: *rID, JobID: j.ID,
				DraftID: rDraftID, Reviewer: derefStr(rReviewer),
				Decision: derefStr(rDecision), EditedText: rEdited, Notes: rNotes,
				CreatedAt: derefTime(rCreatedAt)})
		}
		if pID != nil {
			cur.Published = append(cur.Published, Published{ID: *pID,
				ItemID: derefStr(pItemID), JobID: derefStr(pJobID), DraftID: pDraftID,
				Text: derefStr(pText), ApprovedBy: derefStr(pApprovedBy),
				CreatedAt: derefTime(pCreatedAt)})
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	out := make([]GenerationJob, 0, len(order))
	for _, id := range order {
		out = append(out, *byID[id])
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].CreatedAt.After(out[b].CreatedAt) })
	for i := range out {
		sort.SliceStable(out[i].Drafts, func(a, b int) bool {
			return out[i].Drafts[a].Rank < out[i].Drafts[b].Rank
		})
		sort.SliceStable(out[i].Reviews, func(a, b int) bool {
			return out[i].Reviews[a].CreatedAt.Before(out[i].Reviews[b].CreatedAt)
		})
	}
	return out, nil
}

func derefOr(s *string, def string) string {
	if s == nil {
		return def
	}
	return *s
}

func derefStr(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func derefInt(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}

func derefFloat(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

func derefTime(v *time.Time) time.Time {
	if v == nil {
		return time.Time{}
	}
	return *v
}

func (s *Store) ListDrafts(ctx context.Context, jobID string) ([]Draft, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, job_id, rank, text, score FROM description_drafts
		WHERE job_id=$1 ORDER BY rank`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Draft{}
	for rows.Next() {
		var d Draft
		if err := rows.Scan(&d.ID, &d.JobID, &d.Rank, &d.Text, &d.Score); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func (s *Store) ListReviews(ctx context.Context, jobID string) ([]Review, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, job_id, draft_id, reviewer, decision, edited_text, notes, created_at
		FROM editor_reviews WHERE job_id=$1 ORDER BY created_at`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Review{}
	for rows.Next() {
		var r Review
		if err := rows.Scan(&r.ID, &r.JobID, &r.DraftID, &r.Reviewer, &r.Decision,
			&r.EditedText, &r.Notes, &r.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// ApplyReview records a review and moves the job to nextStatus in one
// transaction, re-checking the current status inside it (SELECT … FOR UPDATE).
//
// Two separate writes used to leave a review row behind with the job still in
// draft_ready when the second failed, which allowed duplicate approvals of the
// same draft and broke the "every transition is audited" rule.
func (s *Store) ApplyReview(ctx context.Context, r Review, nextStatus string) error {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx) //nolint:errcheck

	var current string
	if err := tx.QueryRow(ctx,
		`SELECT status FROM generation_jobs WHERE id=$1 FOR UPDATE`, r.JobID).
		Scan(&current); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNotFound
		}
		return err
	}
	if !workflow.CanReview(current, r.Decision) {
		return fmt.Errorf("%w: cannot %s a job in status %q",
			ErrIllegalTransition, r.Decision, current)
	}
	if err := workflow.AssertTransition(current, nextStatus); err != nil {
		return fmt.Errorf("%w: %v", ErrIllegalTransition, err)
	}
	if _, err := tx.Exec(ctx, `
		INSERT INTO editor_reviews (job_id, draft_id, reviewer, decision, edited_text, notes)
		VALUES ($1,$2,$3,$4,$5,$6)`,
		r.JobID, r.DraftID, r.Reviewer, r.Decision, r.EditedText, r.Notes); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx,
		`UPDATE generation_jobs SET status=$2 WHERE id=$1`, r.JobID, nextStatus); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// ErrIllegalTransition guards every status write.
var ErrIllegalTransition = errors.New("illegal state transition")

// SetJobStatus transitions a description job, refusing illegal moves.
func (s *Store) SetJobStatus(ctx context.Context, id, status string) error {
	tag, err := s.Pool.Exec(ctx, `
		UPDATE generation_jobs SET status=$2
		WHERE id=$1 AND $2 <> status`, id, status)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		// Either the job does not exist, or it is already in the target status.
		var current string
		if err := s.Pool.QueryRow(ctx,
			`SELECT status FROM generation_jobs WHERE id=$1`, id).Scan(&current); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return ErrNotFound
			}
			return err
		}
		return fmt.Errorf("%w: %s -> %s", ErrIllegalTransition, current, status)
	}
	return nil
}

func (s *Store) CreatePublished(ctx context.Context, p Published) (string, error) {
	var id string
	err := s.Pool.QueryRow(ctx, `
		INSERT INTO published_descriptions (item_id, job_id, draft_id, text, approved_by)
		VALUES ($1,$2,$3,$4,$5) RETURNING id`,
		p.ItemID, p.JobID, p.DraftID, p.Text, p.ApprovedBy).Scan(&id)
	return id, err
}

func (s *Store) ListPublishedByJob(ctx context.Context, jobID string) ([]Published, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, item_id, job_id, draft_id, text, approved_by, created_at
		FROM published_descriptions WHERE job_id=$1 ORDER BY created_at`, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Published{}
	for rows.Next() {
		var p Published
		if err := rows.Scan(&p.ID, &p.ItemID, &p.JobID, &p.DraftID, &p.Text,
			&p.ApprovedBy, &p.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *Store) ListPublishedByItem(ctx context.Context, itemID string) ([]Published, error) {
	rows, err := s.Pool.Query(ctx, `
		SELECT id, item_id, job_id, draft_id, text, approved_by, created_at
		FROM published_descriptions WHERE item_id=$1 ORDER BY created_at DESC`, itemID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Published{}
	for rows.Next() {
		var p Published
		if err := rows.Scan(&p.ID, &p.ItemID, &p.JobID, &p.DraftID, &p.Text,
			&p.ApprovedBy, &p.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ------------------------------- platform -----------------------------------

// keyInfo is one row of api_keys.
type keyInfo struct {
	Name string
	Role string
}

// RoleForKeyHash implements httpapi.KeyVerifier against the api_keys table.
//
// An empty role with a nil error means "no such key". A transport/pool failure
// is wrapped so the HTTP layer can answer 503 instead of pretending the caller
// presented a bad key.
func (s *Store) RoleForKeyHash(ctx context.Context, hash string) (string, error) {
	info, err := s.keyInfo(ctx, hash)
	if err != nil {
		return "", err
	}
	return info.Role, nil
}

// NameForKeyHash returns the human-readable key name for audit attribution.
func (s *Store) NameForKeyHash(hash string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	info, err := s.keyInfo(ctx, hash)
	if err != nil || info.Name == "" {
		return "", false
	}
	return info.Name, true
}

func (s *Store) keyInfo(ctx context.Context, hash string) (keyInfo, error) {
	var info keyInfo
	err := s.Pool.QueryRow(ctx,
		`SELECT name, role FROM api_keys WHERE key_hash=$1 AND active`, hash).
		Scan(&info.Name, &info.Role)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return keyInfo{}, nil
	case err != nil:
		if ctx.Err() != nil {
			return keyInfo{}, fmt.Errorf("%w: %v", apperr.ErrAuthUnavailable, ctx.Err())
		}
		return keyInfo{}, fmt.Errorf("%w: %v", apperr.ErrAuthUnavailable, err)
	}
	return info, nil
}

func (s *Store) Audit(ctx context.Context, actor, action, entity, entityID string,
	detail map[string]any) error {
	_, err := s.Pool.Exec(ctx, `
		INSERT INTO audit_log (actor, action, entity, entity_id, detail)
		VALUES ($1,$2,$3,$4,$5)`, actor, action, entity, entityID, marshal(detail))
	return err
}
