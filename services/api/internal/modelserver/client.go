// Package modelserver is the typed client for the Python model server
// (contract: docs/api-contract.md §2).  Timeouts: MODEL_TIMEOUT_SHORT for
// chunk/embed/search, MODEL_TIMEOUT_LONG for generation.  Two retries with
// jittered backoff on 5xx/timeouts.
package modelserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	base   string
	hc     *http.Client
	short  time.Duration
	long   time.Duration
	retryN int
	secret string
}

func New(base string, short, long time.Duration) *Client {
	return &Client{base: strings.TrimRight(base, "/"), short: short, long: long,
		hc: &http.Client{
			Timeout: long + 5*time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        32,
				MaxIdleConnsPerHost: 16,
				IdleConnTimeout:     90 * time.Second,
			},
		}, retryN: 2}
}

// WithSecret attaches the shared secret required by the model server.
func (c *Client) WithSecret(secret string) *Client {
	c.secret = secret
	return c
}

type Chunk struct {
	Ordinal    int    `json:"ordinal"`
	Text       string `json:"text"`
	TokenCount int    `json:"token_count"`
}

type EmbedResponse struct {
	Model        string      `json:"model"`
	ModelVersion string      `json:"model_version"`
	Dim          int         `json:"dim"`
	Vectors      [][]float32 `json:"vectors"`
}

type IndexEntry struct {
	FaissID  int64     `json:"faiss_id"`
	ItemID   string    `json:"item_id"`
	Modality string    `json:"modality"`
	ChunkID  *string   `json:"chunk_id"`
	AssetID  *string   `json:"asset_id"`
	Text     string    `json:"text"`
	Vector   []float32 `json:"vector"`
}

type Hit struct {
	FaissID  int64   `json:"faiss_id"`
	ItemID   string  `json:"item_id"`
	ChunkID  *string `json:"chunk_id"`
	AssetID  *string `json:"asset_id"`
	Modality string  `json:"modality"`
	Score    float64 `json:"score"`
	Text     string  `json:"text"`
}

type SearchResponse struct {
	Hits   []Hit             `json:"hits"`
	Models map[string]string `json:"models"`
	// Degraded is set by the model server when the shared space is untrained,
	// i.e. when cross-modal scores cannot be trusted. It is surfaced to the
	// client instead of being quietly ignored.
	Degraded map[string]any `json:"degraded"`
}

type Citation struct {
	ContextIndex int     `json:"context_index"`
	Score        float64 `json:"score"`
}

type Sentence struct {
	Index     int        `json:"index"`
	Text      string     `json:"text"`
	Citations []Citation `json:"citations"`
}

type CitationCheck struct {
	Passed  bool             `json:"passed"`
	Details []map[string]any `json:"details"`
}

type ComposeContext struct {
	FaissID  *int64  `json:"faiss_id"`
	ItemID   string  `json:"item_id"`
	SKU      *string `json:"sku"`
	ChunkID  *string `json:"chunk_id"`
	AssetID  *string `json:"asset_id"`
	Modality string  `json:"modality"`
	Score    float64 `json:"score"`
	Text     string  `json:"text"`
}

type ComposeResponse struct {
	Answer        string        `json:"answer"`
	Mode          string        `json:"mode"`
	Sentences     []Sentence    `json:"sentences"`
	CitationCheck CitationCheck `json:"citation_check"`
}

type Draft struct {
	Rank  int     `json:"rank"`
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

type GenerateResponse struct {
	Model        string  `json:"model"`
	ModelVersion string  `json:"model_version"`
	Drafts       []Draft `json:"drafts"`
	LatencyMS    int     `json:"latency_ms"`
}

// --------------------------------------------------------------------- //

func (c *Client) doJSON(ctx context.Context, method, path string, in, out any,
	timeout time.Duration) error {
	var body []byte
	if in != nil {
		var err error
		if body, err = json.Marshal(in); err != nil {
			return fmt.Errorf("marshal %s: %w", path, err)
		}
	}
	var lastErr error
	for attempt := 0; attempt <= c.retryN; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<attempt)*150*time.Millisecond +
				time.Duration(rand.Intn(120))*time.Millisecond
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}
		reqCtx, cancel := context.WithTimeout(ctx, timeout)
		req, err := http.NewRequestWithContext(reqCtx, method, c.base+path, bytes.NewReader(body))
		if err != nil {
			cancel()
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		if c.secret != "" {
			req.Header.Set("X-Model-Secret", c.secret)
		}
		resp, err := c.hc.Do(req)
		if err != nil {
			cancel()
			lastErr = fmt.Errorf("model-server %s: %w", path, err)
			continue
		}
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		resp.Body.Close()
		cancel()
		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("model-server %s: %s: %s", path, resp.Status,
				strings.TrimSpace(string(data)))
			continue
		}
		if resp.StatusCode >= 400 {
			return fmt.Errorf("model-server %s: %s: %s", path, resp.Status,
				strings.TrimSpace(string(data)))
		}
		if out != nil {
			return json.Unmarshal(data, out)
		}
		return nil
	}
	return lastErr
}

// Health pings the model-server liveness endpoint.
func (c *Client) Health(ctx context.Context) error {
	return c.doJSON(ctx, http.MethodGet, "/healthz", nil, nil, c.short)
}

func (c *Client) ChunkText(ctx context.Context, text, kind string, maxTokens,
	overlapTokens int) ([]Chunk, error) {
	in := map[string]any{"text": text, "kind": kind,
		"max_tokens": maxTokens, "overlap_tokens": overlapTokens}
	var out struct {
		Chunks []Chunk `json:"chunks"`
	}
	if err := c.doJSON(ctx, http.MethodPost, "/v1/chunk", in, &out, c.short); err != nil {
		return nil, err
	}
	return out.Chunks, nil
}

func (c *Client) EmbedText(ctx context.Context, texts []string) (EmbedResponse, error) {
	var out EmbedResponse
	err := c.doJSON(ctx, http.MethodPost, "/v1/embed/text",
		map[string]any{"texts": texts}, &out, c.short)
	return out, err
}

func (c *Client) EmbedImage(ctx context.Context, assetID, path string) (EmbedResponse, error) {
	var out EmbedResponse
	in := map[string]any{"images": []map[string]any{{"asset_id": assetID, "path": path}}}
	err := c.doJSON(ctx, http.MethodPost, "/v1/embed/image", in, &out, c.short)
	return out, err
}

func (c *Client) UpsertIndex(ctx context.Context, entries []IndexEntry) error {
	return c.doJSON(ctx, http.MethodPost, "/v1/index/upsert",
		map[string]any{"entries": entries}, nil, c.short)
}

func (c *Client) RemoveIndex(ctx context.Context, faissIDs []int64, itemIDs []string) error {
	return c.doJSON(ctx, http.MethodPost, "/v1/index/remove",
		map[string]any{"faiss_ids": faissIDs, "item_ids": itemIDs}, nil, c.short)
}

func (c *Client) SaveIndex(ctx context.Context) error {
	return c.doJSON(ctx, http.MethodPost, "/v1/index/save", map[string]any{}, nil, c.long)
}

func (c *Client) LoadIndex(ctx context.Context) error {
	return c.doJSON(ctx, http.MethodPost, "/v1/index/load", map[string]any{}, nil, c.long)
}

func (c *Client) Search(ctx context.Context, queryText string, queryImagePath *string,
	topK int, itemIDs []string, modality string, fused bool) (SearchResponse, error) {
	in := map[string]any{"top_k": topK, "item_ids": itemIDs,
		"modality": modality, "fused": fused}
	if queryText != "" {
		in["query_text"] = queryText
	}
	if queryImagePath != nil {
		in["query_image"] = map[string]any{"path": *queryImagePath}
	}
	var out SearchResponse
	err := c.doJSON(ctx, http.MethodPost, "/v1/search", in, &out, c.short)
	return out, err
}

func (c *Client) Compose(ctx context.Context, question, mode string,
	contexts []ComposeContext) (ComposeResponse, error) {
	in := map[string]any{"question": question, "mode": mode, "contexts": contexts}
	var out ComposeResponse
	err := c.doJSON(ctx, http.MethodPost, "/v1/answer/compose", in, &out, c.long)
	return out, err
}

func (c *Client) Generate(ctx context.Context, spec map[string]any, beamWidth,
	numReturn, maxLen int) (GenerateResponse, error) {
	in := map[string]any{"spec": spec, "beam_width": beamWidth,
		"num_return_sequences": numReturn, "max_len": maxLen}
	var out GenerateResponse
	err := c.doJSON(ctx, http.MethodPost, "/v1/generate/description", in, &out, c.long)
	return out, err
}
