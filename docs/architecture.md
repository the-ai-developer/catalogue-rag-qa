# Architecture

## Services

```
            ┌──────────────┐        ┌───────────────────────────────┐
 buyers ───▶│  web (SolidJS)│──HTTP──▶│  api (Go :8080)               │
 editors    └──────────────┘        │  · catalogue CRUD + assets    │
                                    │  · auth (X-API-Key, roles)    │
                                    │  · job workers (ingest, gen)  │
                                    │  · QA orchestration           │
                                    └──────┬──────────────┬─────────┘
                                           │ SQL          │ HTTP (internal)
                                    ┌──────▼─────┐  ┌─────▼──────────────┐
                                    │ PostgreSQL │  │ model-server (Py)  │
                                    │ 16         │  │ :8090 · FastAPI    │
                                    └────────────┘  │ · chunker          │
                                                    │ · SBERT + CLIP     │
                                                    │ · shared-space proj│
                                                    │ · FAISS (2 indexes)│
                                                    │ · grounded composer│
                                                    │ · T5 beam search   │
                                                    └─────┬──────────────┘
                                                          │ GPU (CUDA)
                                                    ┌─────▼──────────────┐
                                                    │ 2× NVIDIA T4      │
                                                    │ (trainer: torchrun)│
                                                    └────────────────────┘
```

**Boundaries.** Go owns HTTP, auth, persistence, workflow state machines, audit.
Python owns tokens, vectors, FAISS, generation. Postgres is the source of truth for
catalogue + workflow; FAISS lives under `FAISS_INDEX_DIR` and is both persisted and
rebuildable. Nothing is shown to a user before the citation gate (QA) or editor gate
(descriptions).

**Migrations.** The api owns the schema ledger (`schema_migrations`) and applies
`db/migrations/*.sql` at boot, one transaction per file. Postgres' initdb hook is
deliberately unused so the SQL is never applied twice.

**Internal auth.** The model server's `/v1` routes require `X-Model-Secret` when
`MODEL_SHARED_SECRET` is set, and it only reads files under `ASSET_ROOT`. A DB outage
in the key lookup returns 503, not 401, so an infrastructure failure is never
mistaken for a bad credential.

## Data flows

**Project 1 — Multimodal RAG Q&A**
1. Item specs/titles/copy → `POST /v1/chunk` (token-aware windows, overlap) → `chunks`.
2. Product photos + chunk texts → `/v1/embed/{text,image}` → `ProjectionMapper` →
   one shared space → `embeddings` rows (id = faiss_id) → `/v1/index/upsert`.
   The mapper holds **two** heads (SBERT 768→512, CLIP 512→512) because the
   encoders disagree on width; without trained weights the service reports
   `degraded: untrained_projection` rather than serving meaningless scores.
3. Question → `/v1/search` (both indexes, optional image query) → top-k. Fusion is
   **per item**, not per vector: a text chunk and a photo of the same item are two
   rows with two `faiss_id`s, so they are grouped by `item_id` and combined as
   `text_weight·s_text + image_weight·s_image` before ranking.
4. `/v1/answer/compose` builds the answer **only** from retrieved contexts and emits
   per-sentence citations + a citation check (support score per sentence; any
   unsupported sentence ⇒ `passed=false` and is withheld from display).
5. Query, answer, citations, latency persisted for audit and history.

**Spec to description**
1. Spec sheet captured as structured fields (`spec_sheets`).
2. The model server linearises the spec to one string, then a fine-tuned T5
   encoder-decoder consumes it as subwords.
3. Beam search returns ranked drafts with scores.
4. **Editor gate**: nothing reaches the client until a human approves (or edits and
   approves). Publish writes `published_descriptions` + audit entry.

The T5 weights are trained in a separate repository,
[spec-to-description](https://github.com/the-ai-developer/spec-to-description).
This repository serves the model and enforces the gate; it does not fine-tune it.
The lineariser, `linearise_spec`, is duplicated in both, and
`services/model-server/tests/test_linearise_contract.py` pins the format as exact
strings so the two copies cannot drift apart silently.

## Training

- `python ml/training/train_projection.py` — the two-headed shared-space projection
  (InfoNCE on `(description text, product image)` pairs); single GPU or CPU. This is
  the only training that happens here, and it is what makes cross-modal retrieval
  meaningful at all.
- `ml/notebooks/01_dual_encoder_from_scratch.ipynb` implements the dual encoder and
  its loss in numpy/torch, so the objective can be checked by hand.
  `03_rag_evaluation.ipynb` scores a built index with hit@1 and MRR.
- SBERT and CLIP weights come from HuggingFace. Only the projection between them is
  trained.

## Deployment

- `docker compose up` for local/box: postgres, api, model, web. Set
  `MODEL_DOCKERFILE=deploy/docker/Dockerfile.model-cpu` on a host with no NVIDIA
  GPU. Volumes: `pgdata`, `assets`, `faiss`.
- `deploy/k8s/`: Deployments for api/model/web, GPU `nvidia.com/gpu: "2"` for the
  trainer Job, probes on `/healthz|/readyz`, ConfigMap/Secret for env, PVCs for
  assets/faiss/checkpoints. Model rollouts rebuild/load the FAISS index from disk
  (`/v1/index/load`) and verify `/v1/readyz` before receiving traffic.
- Scaling: api is stateless (HPA on CPU), model-server is GPU-bound (scale by replica
  count; one process per GPU via `CUDA_VISIBLE_DEVICES` sharding if needed).

## Reliability rules

- Every status write goes through a `workflow.CanTransition` guard, and a review
  plus its status change are one transaction (`store.ApplyReview`) — a crash cannot
  leave a review row on a job that is still `draft_ready`.
- Publishing resolves the human-approved text through `workflow.ApprovedText`, which
  returns an error rather than falling back to a raw model draft.
- Audit failures are logged, never discarded with `_ =`.
- Workers claim jobs with `SELECT … FOR UPDATE SKIP LOCKED` (safe with N replicas).
- Model-server calls have timeouts (5s embed/search, 30s generate) + 2 retries with
  jittered backoff on 5xx/timeouts; failures mark jobs `failed` with the error text.
- FAISS is always recoverable: it is persisted (ingest commit + autosave) *and*
  rebuildable by re-running ingest for all active items (idempotent: embeddings are
  upserted by faiss_id, and a batch is validated before anything is written).
- Index writes are atomic (temp file + rename), so a crash mid-save cannot leave a
  truncated index that fails to load.
