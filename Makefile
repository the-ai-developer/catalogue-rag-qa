# catalogue-rag-qa — one-command workflows. `make help` lists targets.
.PHONY: help up down logs psql migrate seed build test test-web test-e2e lint check \
        train-projection ingest-demo e2e-demo notebooks \
        dataset clean fmt vet migrations-check

COMPOSE ?= docker compose
PY ?= python3
API_KEY ?= local-dev-admin-key
N ?= 3200

help:  ## show this help
	@grep -E '^[a-z][a-z0-9-]*:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

up:  ## start postgres + api + model + web (compose)
	$(COMPOSE) up --build -d --wait

down:  ## stop and remove containers
	$(COMPOSE) down

logs:  ## tail all service logs
	$(COMPOSE) logs -f --tail=100

psql:  ## open psql in the postgres container
	$(COMPOSE) exec postgres psql -U catalogue -d catalogue

migrate:  ## apply db/migrations (the api does this on boot; this just restarts it)
	@echo "migrations are applied by the api at startup, tracked in schema_migrations"
	$(COMPOSE) up -d --wait api

seed:  ## seed 20 demo items + ingest + sample traffic (idempotent)
	$(PY) scripts/seed_demo.py --api-base http://localhost:8080 \
		--api-key $(API_KEY)

build:  ## build everything (go + web)
	cd services/api && go build ./...
	cd web && npm run build

# `test` is the fast local loop; `check` is what CI gates on.
test:  ## run all test suites
	cd services/api && go test -count=1 ./...
	cd services/model-server && $(PY) -m pytest tests -q
	cd web && npm run typecheck && npx vitest run

test-web:  ## web unit tests + typecheck (needs `npm ci` once)
	cd web && npm run typecheck && npx vitest run

# Browser-level contract tests. Requires the running stack, because the whole
# point is to exercise nginx -> api and the two product rules in the real UI.
# CHROME_PATH lets you point at a system chromium instead of a downloaded one.
test-e2e:  ## playwright contract tests against the running web origin
	cd web && BASE_URL="$${BASE_URL:-http://127.0.0.1:8081}" \
		API_KEY="$(API_KEY)" npx playwright test

check: migrations-check lint test  ## everything CI runs that needs no stack

migrations-check:  ## fail if SQL is duplicated or mounted into initdb
	@stray=$$(find . -name '*.sql' -not -path './db/migrations/*' -not -path './.git/*'); \
	if [ -n "$$stray" ]; then \
	  echo "db/migrations is the single source of truth; move:"; echo "$$stray"; exit 1; \
	fi
	@# Strip comments first: the compose file documents *why* initdb is unused.
	@if sed 's/#.*$$//' docker-compose.yml | grep -q 'initdb\.d'; then \
	  echo "the api owns schema_migrations; do not mount SQL into initdb.d"; exit 1; \
	fi
	@echo "migrations OK"

lint:  ## go vet + gofmt check + python compile check + compose config
	cd services/api && go vet ./... && test -z "$$(gofmt -l .)"
	$(PY) -m compileall -q services/model-server ml scripts tests
	$(COMPOSE) config -q

fmt:  ## gofmt the api in place
	cd services/api && gofmt -w .

vet:  ## go vet only
	cd services/api && go vet ./...

train-projection:  ## train the two-headed shared-space projection (InfoNCE)
	$(PY) ml/training/train_projection.py --manifest ml/data/generated/manifest.csv \
		--out ml/checkpoints/projection/projection.pt

dataset:  ## regenerate the synthetic training corpus (deterministic)
	$(PY) ml/data/make_synthetic_dataset.py --out ml/data/generated --n $(N) --seed $(SEED)

ingest-demo:  ## trigger ingest for every active item (FAISS rebuild path)
	$(PY) scripts/seed_demo.py --api-base http://localhost:8080 \
		--api-key $(API_KEY) --ingest-only

e2e-demo:  ## full scripted journey with assertions (needs a running stack)
	bash scripts/e2e_demo.sh

notebooks:  ## regenerate notebooks 01 (dual encoder) and 03 (RAG eval)
	$(PY) ml/notebooks/build_notebooks.py

clean:  ## remove build artefacts and generated data (keeps volumes)
	rm -rf web/dist services/api/api services/api/coverage.out \
		ml/data/generated ml/runs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .pytest_cache -type d -prune -exec rm -rf {} +
