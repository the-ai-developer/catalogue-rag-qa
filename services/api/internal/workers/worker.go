// Package workers runs the background pipelines: catalogue ingest (Project 1
// indexing) and description generation (Project 2 beam-search drafts).
package workers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"catalogue-ai/services/api/internal/config"
	"catalogue-ai/services/api/internal/modelserver"
	"catalogue-ai/services/api/internal/store"
	"catalogue-ai/services/api/internal/workflow"
)

type Worker struct {
	Store *store.Store
	MS    *modelserver.Client
	Cfg   *config.Config
}

// Run starts worker loops until ctx is cancelled.
func (w *Worker) Run(ctx context.Context) {
	for i := 0; i < 2; i++ {
		go w.loop(ctx, "ingest", w.ingestOnce)
		go w.loop(ctx, "generation", w.generationOnce)
	}
}

func (w *Worker) loop(ctx context.Context, name string, once func(context.Context) bool) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		if once(ctx) {
			continue // keep draining while jobs are available
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(w.Cfg.WorkerPollInterval):
		}
	}
}

// ------------------------------- ingest -------------------------------------

func (w *Worker) ingestOnce(ctx context.Context) bool {
	job, err := w.Store.ClaimIngestJob(ctx)
	if err != nil {
		// ErrNotFound means "no queued job" — the normal idle case. Anything
		// else is a real fault and was previously indistinguishable from idle,
		// so a broken worker loop looked like an empty queue forever.
		if !errors.Is(err, store.ErrNotFound) {
			slog.Error("claim ingest job failed", "err", err)
		}
		return false
	}
	slog.Info("ingest start", "job", job.ID, "item", job.ItemID)
	chunks, images, err := w.runIngest(ctx, job.ItemID)
	if err != nil {
		msg := err.Error()
		slog.Error("ingest failed", "job", job.ID, "err", msg)
		_ = w.Store.FinishIngestJob(ctx, job.ID, workflow.IngestFailed,
			chunks, images, &msg)
		return true
	}
	_ = w.Store.FinishIngestJob(ctx, job.ID, workflow.IngestSucceeded,
		chunks, images, nil)
	slog.Info("ingest done", "job", job.ID, "chunks", chunks, "images", images)
	return true
}

func (w *Worker) runIngest(ctx context.Context, itemID string) (int, int, error) {
	item, err := w.Store.GetItem(ctx, itemID)
	if err != nil {
		return 0, 0, err
	}

	// idempotent re-ingest: drop old index data first
	oldIDs, err := w.Store.ClearEmbeddings(ctx, itemID)
	if err != nil {
		return 0, 0, err
	}
	if len(oldIDs) > 0 {
		if err := w.MS.RemoveIndex(ctx, oldIDs, []string{itemID}); err != nil {
			return 0, 0, err
		}
	}

	// 1. serialise item fields into source segments
	type segment struct{ kind, text string }
	segs := []segment{}
	if item.Title != "" {
		segs = append(segs, segment{"title", item.Title})
	}
	if item.Material != nil && *item.Material != "" {
		segs = append(segs, segment{"spec", "Material: " + *item.Material})
	}
	if len(item.Dimensions) > 0 {
		b, _ := json.Marshal(item.Dimensions)
		segs = append(segs, segment{"dimension", "Dimensions: " + string(b)})
	}
	for _, f := range item.Features {
		segs = append(segs, segment{"feature", f})
	}
	if len(item.Extra) > 0 {
		b, _ := json.Marshal(item.Extra)
		segs = append(segs, segment{"copy", "Details: " + string(b)})
	}

	// 2. chunk every segment (token-aware, model-server side)
	texts, sources, counts := []string{}, []string{}, []int{}
	for _, seg := range segs {
		chunks, err := w.MS.ChunkText(ctx, seg.text, seg.kind,
			w.Cfg.MaxChunkTokens, w.Cfg.ChunkOverlapTokens)
		if err != nil {
			return 0, 0, err
		}
		for _, c := range chunks {
			texts = append(texts, c.Text)
			sources = append(sources, seg.kind)
			counts = append(counts, c.TokenCount)
		}
	}
	rows, err := w.Store.ReplaceChunks(ctx, itemID, texts, sources, counts)
	if err != nil {
		return 0, 0, err
	}

	// 3. text embeddings → embeddings rows (= faiss ids) → index upsert
	entries := []modelserver.IndexEntry{}
	if len(rows) > 0 {
		emb, err := w.MS.EmbedText(ctx, texts)
		if err != nil {
			return 0, 0, err
		}
		// A count mismatch means the model server truncated or padded the batch.
		// Indexing the first N vectors would attach the wrong text to the wrong
		// chunk, so fail the job instead of corrupting the index silently.
		if len(emb.Vectors) != len(rows) {
			return 0, 0, fmt.Errorf(
				"embedding count %d does not match chunk count %d for item %s",
				len(emb.Vectors), len(rows), itemID)
		}
		for i, row := range rows {
			id, err := w.Store.InsertEmbedding(ctx, itemID, "text", &row.ID, nil,
				emb.Model, emb.ModelVersion, emb.Dim)
			if err != nil {
				return 0, 0, err
			}
			entries = append(entries, modelserver.IndexEntry{
				FaissID: id, ItemID: itemID, Modality: "text", ChunkID: &row.ID,
				Text: row.Text, Vector: emb.Vectors[i]})
		}
	}

	// 4. product photographs → CLIP side of the shared space
	images := 0
	assets, err := w.Store.ListAssets(ctx, itemID)
	if err != nil {
		return 0, 0, err
	}
	for _, a := range assets {
		if a.Kind != "image" {
			continue
		}
		emb, err := w.MS.EmbedImage(ctx, a.ID, a.URL)
		if err != nil {
			return 0, 0, err
		}
		if len(emb.Vectors) != 1 {
			// One image in, one vector out. Anything else means the encoder is
			// misconfigured; skipping it would leave the photo unindexed with
			// the job reported as succeeded.
			return 0, 0, fmt.Errorf("expected 1 image embedding for asset %s, got %d",
				a.ID, len(emb.Vectors))
		}
		id, err := w.Store.InsertEmbedding(ctx, itemID, "image", nil, &a.ID,
			emb.Model, emb.ModelVersion, emb.Dim)
		if err != nil {
			return 0, 0, err
		}
		entries = append(entries, modelserver.IndexEntry{
			FaissID: id, ItemID: itemID, Modality: "image", AssetID: &a.ID,
			Vector: emb.Vectors[0]})
		images++
	}

	// 5. one upsert, then mark indexed
	if err := w.MS.UpsertIndex(ctx, entries); err != nil {
		return 0, 0, err
	}
	faissIDs := make([]int64, 0, len(entries))
	for _, e := range entries {
		faissIDs = append(faissIDs, e.FaissID)
	}
	if err := w.Store.MarkIndexed(ctx, faissIDs); err != nil {
		return 0, 0, err
	}
	// Persist the index to FAISS_INDEX_DIR so a model-server restart does not
	// silently empty retrieval. The model server also autosaves, but the
	// ingest job is the natural commit point.
	if len(entries) > 0 {
		if err := w.MS.SaveIndex(ctx); err != nil {
			slog.Warn("faiss save after ingest failed; index stays in memory "+
				"until the next autosave", "item", itemID, "err", err)
		}
	}
	return len(rows), images, nil
}

// ------------------------------- generation ---------------------------------

func (w *Worker) generationOnce(ctx context.Context) bool {
	job, err := w.Store.ClaimGenerationJob(ctx)
	if err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			slog.Error("claim generation job failed", "err", err)
		}
		return false
	}
	slog.Info("generation start", "job", job.ID, "spec", job.SpecID)
	if err := w.runGeneration(ctx, job); err != nil {
		msg := err.Error()
		slog.Error("generation failed", "job", job.ID, "err", msg)
		_ = w.Store.FinishGenerationJob(ctx, job.ID, workflow.JobFailed,
			nil, nil, &msg)
		return true
	}
	return true
}

func (w *Worker) runGeneration(ctx context.Context, job store.GenerationJob) error {
	spec, err := w.Store.GetSpec(ctx, job.SpecID)
	if err != nil {
		return err
	}
	specMap := map[string]any{
		"category": spec.Category,
	}
	if spec.Title != nil {
		specMap["title"] = *spec.Title
	}
	if spec.Material != nil {
		specMap["material"] = *spec.Material
	}
	if len(spec.Dimensions) > 0 {
		specMap["dimensions"] = spec.Dimensions
	}
	if len(spec.Features) > 0 {
		specMap["features"] = spec.Features
	}
	if len(spec.Extra) > 0 {
		specMap["extra"] = spec.Extra
	}

	res, err := w.MS.Generate(ctx, specMap, job.BeamWidth, 3, job.MaxLen)
	if err != nil {
		return err
	}
	drafts := make([]store.Draft, 0, len(res.Drafts))
	for _, d := range res.Drafts {
		drafts = append(drafts, store.Draft{
			JobID: job.ID, Rank: d.Rank, Text: d.Text, Score: d.Score})
	}
	if err := w.Store.InsertDrafts(ctx, job.ID, drafts); err != nil {
		return err
	}
	model, version := res.Model, res.ModelVersion
	return w.Store.FinishGenerationJob(ctx, job.ID, workflow.JobDraftReady,
		&model, &version, nil)
}
