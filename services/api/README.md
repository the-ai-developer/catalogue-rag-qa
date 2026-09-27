# Go API — `services/api`

Catalogue + workflow backend for both systems. Contract: `docs/api-contract.md §1`.

```bash
cd services/api
go build ./... && go test ./...
go run ./cmd/server            # needs DATABASE_URL + MODEL_SERVER_URL
```

| Package | Responsibility |
| --- | --- |
| `cmd/server` | wiring, graceful shutdown |
| `internal/httpapi` | router, auth (X-API-Key roles), error envelope, handlers |
| `internal/store` | PostgreSQL access (schema: `db/migrations`) |
| `internal/workers` | ingest pipeline + description generation (SKIP LOCKED) |
| `internal/modelserver` | typed model-server client with retries |
| `internal/workflow` | pure state machines (editor gate, job lifecycles) |

**Product rules enforced here:**
1. QA answers persist `citation_check` + per-sentence citations — responses expose
   the check so the UI can withhold ungrounded answers.
2. Editor gate: `review` only from `draft_ready`; `publish` only from `approved`;
   every transition writes `audit_log` in the same transaction.
