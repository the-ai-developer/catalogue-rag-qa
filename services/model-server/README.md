# Python model-server — `services/model-server`

All ML serving: chunking, dual embeddings, shared space, FAISS, grounded
composition, T5 beam search. Contract: `docs/api-contract.md §2`.

```bash
pip install -e ".[ml]"          # GPU serving (torch/transformers/sentence-transformers)
pip install -e ".[dev]"         # tests (CPU-safe, no downloads)
uvicorn app.main:app --host 0.0.0.0 --port 8090

# CPU smoke / tests
EMBEDDER_BACKEND=hash pytest tests -q
```

| Module | Role |
| --- | --- |
| `app/chunker.py` | token-aware overlapping windows |
| `app/embed.py` | SBERT text + CLIP image (L2-normalised), hash fallback |
| `app/projection.py` | learned map into one shared space (identity until trained) |
| `app/faiss_store.py` | dual IndexFlatIP + fused top-k, save/load, sidecar meta |
| `app/answer_composer.py` | extractive/abstractive composition + citation check |
| `app/generator.py` | T5 beam search drafts (template fallback) |
| `app/from_scratch/` | beam search, multi-head attention, InfoNCE (used by notebooks) |
| `app/linearise.py` | **normative** spec → input-string serialisation |

Training lives in `../../ml/` (2×T4 torchrun); notebooks in `../../ml/notebooks/`
rebuild both model families from scratch.
