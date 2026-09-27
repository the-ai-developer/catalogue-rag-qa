# catalogue-rag-qa

Multimodal retrieval and grounded Q&A over an e-commerce catalogue.

Item specs, titles, existing copy and product photographs are chunked and
embedded along two paths, SBERT for text and CLIP for images, then mapped into
one shared space and indexed in FAISS. A buyer asks a question in plain
language, both indexes are searched together, the top-k evidence is kept
readable, and the answer is composed only from what was retrieved. Every
sentence carries a citation, and a per-sentence check runs before anything is
displayed.

If the projection that aligns the two spaces has not been trained, the system
says so: `/readyz` returns 503, every search response carries a `degraded`
block naming the reason, and the Ask page shows a banner. It does not serve
hash noise as if it were a shared space.

```
catalogue-rag-qa/
├── services/
│   ├── api/                    Go 1.22: catalogue, ingest workers, QA, editor gate
│   └── model-server/           Python FastAPI: chunking, embeddings, FAISS, composer
├── web/                        SolidJS: Catalogue, Item, Ask, Descriptions
├── db/migrations/              PostgreSQL schema, single source of truth
├── ml/
│   ├── contrastive.py          InfoNCE loss, the training objective
│   ├── training/               projection fine-tune
│   ├── eval/                   retrieval scoring
│   └── notebooks/              01 dual encoder, 03 RAG evaluation
├── deploy/                     Dockerfiles (GPU and CPU), k8s manifests
├── scripts/                    seed_demo.py, e2e_demo.sh
└── docs/                       architecture, API contract, operations runbook
```

## Quickstart

```bash
cp .env.example .env     # the api refuses to start on a placeholder secret
make up                  # postgres + api + model + web, waits for health
make seed                # 20 demo items, ingest, sample traffic
make e2e-demo            # scripted journey with assertions
# web :8081  ·  api :8080  ·  model :8090
```

No NVIDIA GPU? Put `MODEL_DOCKERFILE=deploy/docker/Dockerfile.model-cpu` in
`.env`. You get a small model server running the hash embedder, which is enough
to exercise the whole stack and not enough to trust the vectors. It reports
itself as degraded.

`make seed` is idempotent. Add `--reset` to start over or `--no-images` to skip
the slower image indexing pass.

The Python side needs a virtualenv on distributions that mark the system Python
externally managed (PEP 668), which is most of them now:

```bash
python3 -m venv .venv && . .venv/bin/activate
```

`make check` and the model-server tests run without installing anything, since
the tests put the repository on `sys.path` themselves. That is deliberate: the
suite should work on a fresh checkout with one command and no setup.

Ask it something:

```bash
curl -s localhost:8080/api/v1/qa/ask \
  -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Is the Kestrel bottle dishwasher safe?","top_k":6}'
```

## What the system refuses to do

Six rules, all enforced in code and covered by tests.

1. **Grounded or nothing.** Answers carry per-sentence citations. The UI renders
   no answer text when the citation check fails.
2. **The editor gate holds.** Generated descriptions are drafts. Publishing
   resolves approved text through `workflow.ApprovedText`, which returns an
   error rather than falling back to the raw model output.
3. **Indexes are persisted and disposable.** FAISS is written on every ingest
   commit and on a timer, and can be wiped and rebuilt from Postgres and the
   asset volume at any time.
4. **Transitions are transactional.** Every job change passes
   `workflow.CanTransition`. A review and its status change commit together.
   Every transition is audited with the authenticated key's name, and a failed
   audit is logged rather than dropped.
5. **Degradation is announced.** An untrained projection, a key-store outage or
   a missing checkpoint each surface as a 503, a `degraded` block, or a banner.
6. **Failures are visible.** A failed load renders its error instead of an
   eternal spinner. A 500 logs its cause.

## Two things the tests were built around

The worst defects in this codebase were invisible to unit tests. All three of
these passed `go test`, `pytest` and `vitest` and only appeared when the real
images were built and a real browser loaded them:

- The Vite bundle was emitted to `dist/assets/`, which nginx had aliased to the
  product-photo volume, so the app's own JavaScript 404'd and the SPA rendered
  a blank page. `GET /` still returned valid HTML, which is why an earlier
  check passed. Fixed with `assetsDir: 'static'`, and CI fails if `dist/index.html`
  ever references `/assets/` again.
- `GET /api/v1/items` filtered on `text <% text`, an operator Postgres does not
  have, so the catalogue page failed outright.
- A stored jsonb `null` in `features` reached the browser as `null` rather than
  `[]`, `features.join()` threw, and the error banner was nested inside the
  loading branch, so the page spun forever.

The Playwright suite exists for that reason. It is not decoration.

## Tests

```bash
make check                 # everything CI runs that needs no stack
make test                  # go test + pytest + tsc + vitest
cd web && npm ci           # once; the lockfile pins registry.npmjs.org
cd web && npx playwright test   # needs `make up`
```

```
services/api      7 packages, race detector on
services/model-server  90 tests
web               45 unit tests + 12 browser contract tests
tests/integration 18 live tests against a real database
```

## Training the shared space

Cross-modal retrieval is meaningless until the two spaces are aligned.

```bash
make dataset           # deterministic synthetic corpus: specs, images, qrels
make train-projection  # two-headed projection, InfoNCE, single GPU or CPU
```

SBERT 768 and CLIP 512 go in, both land at 512. The objective is InfoNCE over
paired `(description text, product image)`, and it lives in
[`ml/contrastive.py`](ml/contrastive.py) rather than in a notebook, because it is
the function the whole retrieval story depends on. It has 13 tests, including
one that pins the loss against a hand-computed cross entropy and one that fails
if only one tower is updated.

Until it has run, the stack says degraded rather than pretending.

## Relationship to spec-to-description

[`spec-to-description`](https://github.com/the-ai-developer/spec-to-description)
is the other half: the T5 that turns specs into copy, with the attention and
beam search written out by hand. This repository serves those drafts and gates
them behind an editor. The two are independent and neither imports the other at
runtime, though `linearise_spec` is deliberately duplicated between them and
pinned by an identical test in both.

## Stack

Go 1.22, SolidJS, PostgreSQL with pgvector, Python with FAISS, SBERT and CLIP,
nginx. Docker Compose for local, k8s manifests for deployment.

## License

MIT. See [LICENSE](LICENSE).
