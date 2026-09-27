# Operations runbook

## First deploy

```bash
cp .env.example .env          # set BOOTSTRAP_API_KEYS + MODEL_SHARED_SECRET
make up                       # postgres + api + model + web, waits for health
make seed                     # demo catalogue + traffic (idempotent)
make e2e-demo                 # assert the two product rules end-to-end
```

### No GPU? Use the CPU model image

The default model image is a ~3.8 GB CUDA build. On any machine without an
NVIDIA GPU (and on CI) select the small CPU image instead:

```bash
# in .env
MODEL_DOCKERFILE=deploy/docker/Dockerfile.model-cpu
```

It has no torch/transformers and runs with `EMBEDDER_BACKEND=hash`, so vectors
are meaningless — the model server reports `degraded: untrained_projection` and
`/readyz` returns 503 until you train a real projection. That is intentional:
better a loud 503 than confident nonsense.

Services: web `:8081` · api `:8080` · model `:8090` · postgres `:5432`.

### Who applies migrations

The **api** owns the schema. It applies `db/migrations/*.sql` on boot inside a
transaction each and records every file in `schema_migrations`.

Postgres' `docker-entrypoint-initdb.d` is deliberately **not** used. It would
apply the same SQL outside that ledger, and the api would then abort on
`relation "items" already exists` on every fresh volume. `make migrations-check`
and the `hygiene` CI job both fail if SQL is duplicated or mounted into initdb.

```bash
docker compose up -d --wait api    # = `make migrate`
```

### Bootstrap API keys

`BOOTSTRAP_API_KEYS=name:sha256hex:role[,…]` seeds keys at first boot; the
`api_keys` table is authoritative afterwards. Generate a hash:

```bash
python3 -c "import hashlib;print(hashlib.sha256(b'MY-SECRET').hexdigest())"
```

Roles: `viewer` (read + ask) < `editor` (review/publish) < `admin` (catalogue + ingest).

Resolved roles are memoised for `AUTH_CACHE_TTL` (30s), so a revoked key keeps
working for at most that long.

**The api refuses to boot** while any secret still contains `CHANGE_ME` or
`REPLACE_WITH_SHA256_HEX`. A placeholder in `deploy/k8s/k8s.yaml` is a valid
credential whose secret is published in this repository.

### The model server's shared secret

Every `/v1` route requires `X-Model-Secret: $MODEL_SHARED_SECRET` once that
variable is set. Leave it empty only for local dev. The model server reads
files under `ASSET_ROOT` on request, so an unauthenticated one is a
file-read primitive — path traversal is blocked, but auth is the real control.

## Backups (Postgres = system of record)

```bash
make psql
pg_dump -U catalogue -Fc catalogue > backup.dump       # nightly via cron/CI
pg_restore -U catalogue -d catalogue --clean backup.dump
```

Back up alongside the DB: `/data/assets` (product photos) and
`/data/faiss` (the persisted index).

## URL spaces in the web image (do not merge these)

| Path      | Served by                                        |
| --------- | ------------------------------------------------ |
| `/api/`   | nginx → Go api (reverse proxy)                   |
| `/readyz` | nginx → Go api                                   |
| `/healthz`| the web container itself (its own liveness probe) |
| `/assets/`| the shared product-photo volume                  |
| `/static/`| the built SPA bundle                             |
| `/`       | the SPA, with history fallback                   |

`/static` is Vite's `build.assetsDir` and it must not be `assets`: with the
default, the app's own JS sat behind the product-photo alias, 404'd, and the SPA
rendered a blank page. CI fails the build if `dist/index.html` references
`/assets/`.

## Volume ownership

| Volume    | Written by | uid    |
| --------- | ---------- | ------ |
| `assets`  | api        | 65532  |
| `faiss`   | model      | 10001  |
| `pgdata`  | postgres   | 70     |

Named volumes are created root-owned, so each image declares its own directory
with the right ownership. The model container deliberately does **not** chown
`/data/assets` — it only reads it, and the api owns the writes. If photo upload
starts failing with "asset storage is not writable", a stale volume is the
cause:

```bash
docker volume rm <project>_assets && docker compose up -d
```

The api logs a warning at boot if the directory is not writable, so this shows up
before the first upload rather than as a 500.

## The FAISS index

The index **is** persisted. It is written on every ingest commit and by an
autosave thread every `FAISS_AUTOSAVE_SECS`, and reloaded at model-server
startup. A restart no longer empties retrieval.

```bash
curl localhost:8090/readyz | jq .index          # counts actually on disk
curl -X POST localhost:8090/v1/index/save       # force a write now
curl -X POST localhost:8090/v1/index/load       # re-read from disk
```

If it is ever wrong or lost, it is rebuildable from Postgres + assets:

```bash
docker compose exec model rm -rf /data/faiss/*  # wipe
docker compose restart model                    # comes up empty + reports it
make ingest-demo                                # idempotent replay
```

`FAISS_AUTOSAVE_SECS=0` disables persistence. Do that only for throwaway runs.

## Shared-space projection (read this before trusting a cross-modal answer)

SBERT emits 768 dims and CLIP 512, so the projection holds **two** bias-free
matrices (768→512 and 512→512) trained jointly with InfoNCE.

Without those weights there is no shared space: text and photo vectors are not
aligned and cross-modal scores are noise. The server does **not** hide this —
`/readyz`, `/v1/models` and every `/v1/search` response carry
`degraded: {reason: "untrained_projection", …}`, the Ask page shows a banner, and
`/readyz` returns 503 so Kubernetes stops routing traffic.

```bash
make dataset               # deterministic synthetic corpus
make train-projection      # → ml/checkpoints/projection/projection.pt
docker compose restart model && curl localhost:8090/readyz | jq .degraded
```

## Rolling out generator weights

The T5 checkpoint is trained in
[spec-to-description](https://github.com/the-ai-developer/spec-to-description),
not here. This repository only serves it, so a rollout is a file copy plus a
restart.

```bash
# in the spec-to-description checkout
make train                            # → checkpoints/t5-description
rsync -a checkpoints/t5-description/ <this-repo>/ml/checkpoints/t5-description/
docker compose restart model
curl localhost:8090/v1/models | jq .generator   # confirm 'fine-tuned'
# smoke: generate a draft, confirm the editor gate still blocks auto-publish
curl -s -X POST localhost:8090/v1/generate/description \
  -H 'Content-Type: application/json' \
  -H "X-Model-Secret: $MODEL_SHARED_SECRET" \
  -d '{"spec":{"category":"drinkware","material":"18/8 steel"}}'
```

`model_version` in the response is the source of truth: `…:fine-tuned`,
`…:base-not-fine-tuned` (no checkpoint found, base T5 loaded),
`template-fallback:…` (transformers missing), or `load-failed:…` (a checkpoint
exists but will not load — that one is a hard failure, not a fallback).

Keep the previous checkpoint: rollback = restore files + restart.

## GPU health

```bash
nvidia-smi
docker compose exec model python -c "import torch;print(torch.cuda.device_count())"
```

Degraded: `MODEL_DEVICE=cpu` keeps serving (slower). `EMBEDDER_BACKEND=hash` is
for tests only — vectors are meaningless and `/readyz` says so.

## Scaling notes

- **api**: stateless; scale freely (workers claim with `FOR UPDATE SKIP LOCKED`).
- **model**: GPU-bound; one process per GPU, shard by `CUDA_VISIBLE_DEVICES`.
  **The FAISS index is per-process**, so replicas do not share vectors — scale
  by sharding items across replicas, not by adding replicas behind one index.
- **postgres**: managed service in prod; pool is `10` per replica
  (`2 × api replicas × 10` connections total).

## Incident triage

| Symptom | Check | Fix |
| --- | --- | --- |
| `/readyz` 503 `model_server` | model logs, `curl :8090/healthz` | restart model |
| `/readyz` 503 `untrained_projection` | `/v1/models` → `projection.trained` | `make train-projection`, restart model |
| Answers always `passed=false` | `retrieval.degraded` in the response | as above |
| Cross-modal results look random | `TEXT_INDEX_WEIGHT` / `matched_modalities` | retrain the projection; fusion is per item |
| Ingest jobs stuck `queued` | api logs | workers claim via SKIP LOCKED — restart api |
| Ingest `failed` with "embedding count N does not match chunk count M" | model logs | encoder/projection dimension drift |
| Generation `failed` | job `error` column | model OOM? lower `beam_width`/`max_len` |
| Publish 409 | job status | by design: only `approved` jobs publish |
| Publish 409 "no human-approved text" | `editor_reviews` for the job | the job was approved without a review row; re-review it. Publishing is never allowed to fall back to a raw draft |
| 503 on every authenticated call | api logs `key store unavailable` | database is down — this is deliberately **not** a 401, so nobody is silently signed out |
| 401 on a valid key | key revoked, or `AUTH_CACHE_TTL` not elapsed | wait out the TTL or restart api |
| Product photos 404 | the stored `url` must start with `/assets/`; `Vite assetsDir` must not be `assets` | both |
| Photo upload 503 | api logs "asset storage is not writable" | recreate the assets volume (see ownership above) |
| Item page stuck on "Loading item…" | the error banner now renders outside the loading gate; check the browser console | was a null `features` array crashing the render |
| SPA renders blank | `curl localhost:8081/` and fetch the `/static/*.js` it references | a 404 there means a build/alias collision |
| Every API call 404s from the browser | nginx `location /api/` proxy_pass | the SPA is same-origin by design |
