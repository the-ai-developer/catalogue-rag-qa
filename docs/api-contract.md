# API Contract (single source of truth)

All services speak JSON, UTF-8, RFC 3339 timestamps. This document is normative:
`services/api` (Go), `services/model-server` (Python) and `web` (SolidJS) must match it.

## Conventions

- Error envelope (any non-2xx):

```json
{ "error": { "code": "not_found", "message": "item not found", "details": {} } }
```

  Codes: `bad_request`, `unauthorized`, `forbidden`, `not_found`, `conflict`,
  `unprocessable`, `upstream`, `internal`.
- Pagination: `?limit=<1..100, default 20>&cursor=<opaque>`; response
  `{ "items": [...], "next_cursor": "<opaque|null>" }`.
- IDs are UUID strings unless stated otherwise.

---

## 1. Go API — `http://api:8080` (public, prefix `/api/v1`)

Auth: header `X-API-Key: <secret>`. Roles: `viewer` < `editor` < `admin`.
`GET /healthz`, `GET /readyz`, `GET /metrics` (Prometheus) are unauthenticated.

| Method & path | Role | Purpose |
| --- | --- | --- |
| `POST /api/v1/items` | admin | create item |
| `GET /api/v1/items` | viewer | list/search (`query`, `category`, `status`) |
| `GET /api/v1/items/{id}` | viewer | item + assets + chunk summary |
| `PATCH /api/v1/items/{id}` | admin | partial update |
| `DELETE /api/v1/items/{id}` | admin | soft delete (sets `status=archived`, `deleted_at`) |
| `POST /api/v1/items/{id}/assets` | admin | multipart `file` (png/jpg/webp), stores under `ASSET_STORAGE_DIR` |
| `POST /api/v1/items/{id}/ingest` | admin | enqueue chunk+embed+index; returns job |
| `GET /api/v1/jobs/ingest/{id}` | viewer | ingest job status |
| `POST /api/v1/qa/ask` | viewer | multimodal RAG answer (Project 1) |
| `GET /api/v1/qa/answers/{id}` | viewer | stored answer + citations |
| `GET /api/v1/qa/history` | viewer | recent Q&A (`user_ref` filter) |
| `POST /api/v1/descriptions/jobs` | editor | create spec + generation job |
| `GET /api/v1/descriptions/jobs/{id}` | viewer | job + drafts |
| `GET /api/v1/descriptions/jobs` | viewer | list jobs by status |
| `POST /api/v1/descriptions/jobs/{id}/review` | editor | editor gate: approve/reject/edit |
| `POST /api/v1/descriptions/jobs/{id}/publish` | editor | publish approved text to item |
| `GET /api/v1/descriptions/published/{item_id}` | viewer | published history |

### 1.1 Item shapes

```jsonc
// POST /api/v1/items  → 201 Item
{
  "sku": "KX-1042",
  "title": "Kestrel 12 oz Insulated Bottle",
  "category": "drinkware",
  "material": "18/8 stainless steel",
  "dimensions": { "height_cm": 26.0, "diameter_cm": 7.2, "capacity_oz": 12 },
  "features": ["double-wall vacuum insulation", "leak-proof lid", "BPA-free"],
  "extra": { "colours": ["slate", "sand"] },
  "status": "draft"
}
// Item = { id, sku, title, category, material, dimensions, features, extra,
//          status, created_at, updated_at,
//          assets: [Asset], counts: { chunks: n, embeddings_text: n, embeddings_image: n } }
// Asset = { id, kind, url, mime, width, height, bytes, sha256, created_at }
```

`PATCH` accepts any subset of `title, category, material, dimensions, features, extra, status`.

### 1.2 Ingest

`POST /api/v1/items/{id}/ingest` → `202 { "job_id": "...", "status": "queued" }`.
Worker pipeline: serialise item fields → `POST /v1/chunk` (model-server) → store chunks →
`POST /v1/embed/text` + `/v1/embed/image` → allocate `embeddings.id` →
`POST /v1/index/upsert` → mark `indexed_at`. `GET /api/v1/jobs/ingest/{id}` returns
`IngestJob = { id, item_id, status, chunks_indexed, images_indexed, error, created_at, started_at, finished_at }`.

### 1.3 QA — Project 1

`POST /api/v1/qa/ask`

```jsonc
// request
{
  "question": "Is the Kestrel bottle dishwasher safe and what is its capacity?",
  "item_ids": ["optional scope; empty = whole catalogue"],
  "top_k": 6,
  "use_images": true,
  "composition": "extractive" | "abstractive",   // default extractive
  "user_ref": "buyer:acme"
}
// 200 response
{
  "answer_id": "...",
  "answer": "…grounded answer…",
  "composition": "extractive",
  "citations": [
    { "sentence_index": 0, "item_id": "...", "sku": "KX-1042", "chunk_id": "...",
      "asset_id": null, "modality": "text", "score": 0.83,
      "snippet": "…top matching evidence span…" }
  ],
  "citation_check": { "passed": true, "unsourced_sentences": [] },
  "retrieval": { "text_hits": 4, "image_hits": 2, "models": { "text": "…", "image": "…" } },
  "latency_ms": 412
}
```

Server behaviour: retrieve via model-server `/v1/search` (text + image indexes, fused),
compose via `/v1/answer/compose`, persist query/answer/citations. **Never display an
answer whose `citation_check.passed` is false** — such answers are stored with
`citation_check_passed=false` and returned with the flag so the UI can gate on it.

### 1.4 Descriptions

```jsonc
// POST /api/v1/descriptions/jobs → 202
{
  "spec": {
    "item_id": "optional",
    "title": "Kestrel 12 oz Insulated Bottle",
    "category": "drinkware",
    "material": "18/8 stainless steel",
    "dimensions": { "height_cm": 26.0, "capacity_oz": 12 },
    "features": ["double-wall vacuum insulation", "leak-proof lid"],
    "extra": {}
  },
  "beam_width": 4,
  "max_len": 192
}
// response { "job_id": "...", "status": "queued" }

// GET /api/v1/descriptions/jobs/{id}
{ "id": "...", "spec": { …SpecSheet… }, "status": "draft_ready",
  "model_name": "t5-small-desc", "model_version": "…", "beam_width": 4,
  "drafts": [ { "id": "...", "rank": 1, "text": "…", "score": -0.42 } ],
  "reviews": [ … ], "published": [ … ] }

// POST /api/v1/descriptions/jobs/{id}/review  → 200 job
{ "decision": "approve" | "reject" | "edit",
  "draft_id": "required for approve/edit",
  "edited_text": "required for edit",
  "notes": "optional" }

// POST /api/v1/descriptions/jobs/{id}/publish → 201
{ "id": "published_description_id", "item_id": "...", "text": "…", "approved_by": "…" }
```

State machine: `queued → running → draft_ready → (approve|edit → approved | reject → rejected) → published`.
`edit` stores the human-edited text in `editor_reviews.edited_text` and publishes that.
Only `approved` jobs can be published. Every transition is written to `audit_log`.

---

## 2. Model server — `http://models:8090` (internal, prefix `/v1`)

No auth (cluster-internal); bind to pod network only. `GET /healthz`, `GET /readyz`,
`GET /v1/models` (names, versions, dims, device).

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chunk` | token-aware chunking |
| `POST /v1/embed/text` | SBERT text embeddings |
| `POST /v1/embed/image` | CLIP image embeddings |
| `POST /v1/index/upsert` | add/replace vectors in FAISS |
| `POST /v1/index/remove` | delete by `faiss_ids` or `item_ids` |
| `POST /v1/search` | fused top-k over both indexes |
| `POST /v1/answer/compose` | grounded answer + citation check |
| `POST /v1/generate/description` | beam-search spec → description drafts |
| `POST /v1/index/save`, `POST /v1/index/load` | persist/restore FAISS to `FAISS_INDEX_DIR` |

```jsonc
// POST /v1/chunk
{ "item_id": "…", "kind": "spec", "text": "…", "max_tokens": 220, "overlap_tokens": 40 }
// → { "chunks": [ { "ordinal": 0, "text": "…", "token_count": 187 } ] }

// POST /v1/embed/text  →  { "model": "all-mpnet-base-v2", "model_version": "…", "dim": 768, "vectors": [[…]] }
// POST /v1/embed/image
{ "images": [ { "asset_id": "…", "path": "/data/assets/…png" } ] }   // or "b64"
// → { "model": "clip-vit-b32-proj", "model_version": "…", "dim": 512, "vectors": [[…]] }

// POST /v1/index/upsert
{ "entries": [ { "faiss_id": 1042, "item_id": "…", "modality": "text",
                 "chunk_id": "…", "asset_id": null, "vector": [ … ] } ] }
// → { "upserted": 1 }

// POST /v1/search
{ "query_text": "dishwasher safe?", "query_image": { "path": "…" } ,  // either/both
  "top_k": 6, "item_ids": [], "modality": "any",
  "fused": true }
// → { "hits": [ { "faiss_id": 1042, "item_id": "…", "chunk_id": "…", "asset_id": null,
//                 "modality": "text", "score": 0.83, "text": "…chunk text…" } ],
//     "models": { "text": "…", "image": "…" } }

// POST /v1/answer/compose
{ "question": "…", "mode": "extractive" | "abstractive",
  "contexts": [ { "faiss_id": 1042, "item_id": "…", "chunk_id": "…", "asset_id": null,
                  "modality": "text", "score": 0.83, "text": "…", "sku": "KX-1042" } ] }
// → { "answer": "…", "mode": "extractive",
//     "sentences": [ { "index": 0, "text": "…", "citations": [ {"context_index": 0, "score": 0.88} ] } ],
//     "citation_check": { "passed": true,
//       "details": [ { "sentence_index": 1, "supported": true, "support_score": 0.77 } ] } }

// POST /v1/generate/description
{ "spec": { "title": "…", "category": "…", "material": "…",
            "dimensions": {…}, "features": [ … ], "extra": {…} },
  "beam_width": 4, "num_return_sequences": 3, "max_len": 192 }
// → { "model": "t5-small-desc", "model_version": "…",
//     "drafts": [ { "rank": 1, "text": "…", "score": -0.42 } ],
//     "latency_ms": 240 }
```

### Shared-space mapping (normative)

SBERT and CLIP raw outputs live in different spaces. Both paths are L2-normalised and
then mapped by `ProjectionMapper` (learned linear map + re-normalise, trained with
InfoNCE on item-level `(text, image)` pairs — see `ml/training/train_projection.py`)
into one `EMBED_DIM`-dimensional shared space. `/v1/search` scores are inner products
(== cosine) in that space; fused scoring uses weighted sum
`w_text * s_text + w_image * s_image` (default 0.6/0.4) over the retrieved union.

### Spec linearisation (normative, duplicated in Python only)

```
spec: category=drinkware | material=18/8 stainless steel |
      dimensions: height_cm=26.0, capacity_oz=12 |
      features: double-wall vacuum insulation, leak-proof lid | title: Kestrel 12 oz …
```

Fields are emitted in fixed order (category, material, dimensions sorted by key,
features in given order, title, extra sorted by key); missing fields are omitted.
Python owns this function (`model_server/linearise.py`); Go sends structured spec only.
