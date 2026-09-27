# Cross-service integration tests

Black-box tests against a **live** stack. They enforce the two product rules
end-to-end, not just unit behaviour:

1. **Grounded or nothing** — `/api/v1/qa/ask` must return `citation_check` and
   per-sentence citations; ungrounded answers must be flagged (`passed=false`).
2. **Editor gate** — publishing a description before approval must fail; only
   `approve`/`edit`-approved jobs can be published.

```bash
make up && make migrate && make seed
python3 -m pytest tests/integration -q        # or:
python3 tests/integration/test_full_flow.py   # stdlib only, no pytest needed

# custom target / keys
API_BASE=http://box:8080 API_KEY=... python3 -m pytest tests/integration -q
```

No third-party dependencies — pure stdlib (urllib + unittest), so it runs
anywhere, including CI service containers and the `make e2e-demo` path.
