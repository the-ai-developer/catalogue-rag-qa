"""Catalogue AI model-server (FastAPI) — contract: docs/api-contract.md §2.

Serves chunking, dual embeddings (SBERT + CLIP → one shared space), FAISS search,
grounded answer composition with citation checks, and T5 beam-search drafts.
Heavy models load lazily; ``EMBEDDER_BACKEND=hash`` gives a deterministic
CPU-only mode for tests and smoke runs.

Two invariants this module enforces rather than assumes:

* **Authentication.** When ``MODEL_SHARED_SECRET`` is set, every ``/v1`` route
  requires a matching ``X-Model-Secret``. The container is otherwise an
  unauthenticated read/write door onto the filesystem and the index.
* **Degradation is visible.** Without trained projection weights the two towers
  are not aligned and cross-modal recall is noise. That is reported by
  ``/readyz``, by ``/v1/models`` and by every ``/v1/search`` response instead of
  being served as if it were a working shared space.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .answer_composer import AnswerComposer
from .chunker import Tokenizer, chunk_text
from .config import get_settings
from .embed import build_encoders
from .faiss_store import FaissStore
from .generator import DescriptionGenerator
from .projection import ProjectionMapper

log = logging.getLogger(__name__)

# Max accepted request bodies. Without these, /v1/index/upsert will happily
# accept a multi-GB JSON array and OOM the box.
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_EMBED_BATCH = 512


# --------------------------------------------------------------------------- #
# request / response models (shapes == api-contract.md §2)
# --------------------------------------------------------------------------- #


class ChunkReq(BaseModel):
    item_id: Optional[str] = None
    kind: str = "copy"
    text: str
    max_tokens: int = Field(220, ge=8, le=4096)
    overlap_tokens: int = Field(40, ge=0, le=2048)


class EmbedTextReq(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=MAX_EMBED_BATCH)
    model: Optional[str] = None


class ImagePayload(BaseModel):
    asset_id: Optional[str] = None
    path: Optional[str] = None
    b64: Optional[str] = None


class EmbedImageReq(BaseModel):
    images: List[ImagePayload] = Field(min_length=1, max_length=64)


class IndexEntry(BaseModel):
    faiss_id: int
    item_id: str
    modality: str
    chunk_id: Optional[str] = None
    asset_id: Optional[str] = None
    text: str = ""
    vector: List[float]


class UpsertReq(BaseModel):
    entries: List[IndexEntry] = Field(min_length=1)


class RemoveReq(BaseModel):
    faiss_ids: Optional[List[int]] = None
    item_ids: Optional[List[str]] = None


class SearchReq(BaseModel):
    query_text: Optional[str] = None
    query_image: Optional[ImagePayload] = None
    top_k: int = Field(6, ge=1, le=100)
    item_ids: Optional[List[str]] = None
    modality: str = "any"
    fused: bool = True


class ComposeContext(BaseModel):
    faiss_id: Optional[int] = None
    item_id: str
    sku: Optional[str] = None
    chunk_id: Optional[str] = None
    asset_id: Optional[str] = None
    modality: str = "text"
    score: float = 0.0
    text: str = ""


class ComposeReq(BaseModel):
    question: str
    mode: str = "extractive"
    contexts: List[ComposeContext] = Field(default_factory=list)


class GenerateReq(BaseModel):
    spec: Dict[str, Any]
    beam_width: int = Field(4, ge=1, le=32)
    num_return_sequences: int = Field(3, ge=1, le=16)
    max_len: int = Field(192, ge=16, le=512)


# --------------------------------------------------------------------------- #
# runtime state (lazy)
# --------------------------------------------------------------------------- #


class Runtime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.encoders = None
        self.projection = None
        self.store = None
        self.generator = None
        self.tokenizer = None
        self._autosave: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def init(self) -> None:
        s = self.settings
        self.encoders = build_encoders(s)
        self.projection = ProjectionMapper.load(s.projection_path, s.embed_dim)
        self.store = FaissStore(s.embed_dim, s.faiss_index_dir)
        loaded = self.store.load()  # no-op when nothing persisted yet
        self.generator = DescriptionGenerator(s.desc_model_dir, s.model_device,
                                              s.model_seed)
        # A real subword tokenizer so MAX_CHUNK_TOKENS means tokens, not words.
        self.tokenizer = Tokenizer(s.text_model_name)
        self._start_autosave()
        log.info("model-server ready: %s | projection=%s | index=%s | tokenizer=%s",
                 self.encoders.models(), self.projection.info(),
                 self.store.counts(), self.tokenizer.name)

    # ------------------------------------------------------------------ #
    def _start_autosave(self) -> None:
        """Flush the index periodically so a restart does not lose it.

        Without this the index only ever exists in memory: every model restart
        silently empties retrieval until someone re-runs ingest for every item.
        """
        interval = self.settings.faiss_autosave_secs
        if interval <= 0:
            log.info("faiss autosave disabled; index will be lost on restart")
            return

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    if self.store.save_if_dirty(interval):
                        log.info("faiss autosaved to %s", self.store.index_dir)
                except Exception:  # noqa: BLE001 - autosave must never kill the loop
                    log.exception("faiss autosave loop error")

        self._autosave = threading.Thread(target=loop, name="faiss-autosave",
                                          daemon=True)
        self._autosave.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self.store is not None and self.store.dirty:
            try:
                self.store.save()
                log.info("faiss index saved on shutdown")
            except OSError as exc:
                log.error("could not save faiss index on shutdown: %s", exc)

    # ------------------------------------------------------------------ #
    def degraded(self) -> Optional[dict]:
        """Report an unusable state instead of serving meaningless results."""
        if self.projection is None or not self.projection.trained:
            return {
                "reason": "untrained_projection",
                "detail": "SBERT and CLIP outputs are not aligned; cross-modal "
                          "scores are meaningless. Run ml/training/train_projection.py.",
                "path": self.settings.projection_path,
            }
        if self.projection.legacy:
            return {
                "reason": "legacy_single_matrix_projection",
                "detail": "one matrix is applied to both towers; retrain for a "
                          "true shared space.",
                "path": self.settings.projection_path,
            }
        return None

    def encode_text(self, texts: List[str]) -> np.ndarray:
        return self.projection.project_text(self.encoders.text.encode(texts))

    def encode_image(self, images: List[bytes]) -> np.ndarray:
        return self.projection.project_image(self.encoders.image.encode(images))


RT = Runtime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    RT.init()
    try:
        yield
    finally:
        RT.shutdown()


app = FastAPI(title="catalogue-ai model-server", version="1.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #


def require_secret(x_model_secret: Optional[str] = Header(default=None)) -> None:
    """Constant-time check of the shared secret; open when unset (dev only)."""
    expected = RT.settings.shared_secret
    if not expected:
        return
    if not x_model_secret or not hmac.compare_digest(x_model_secret, expected):
        raise HTTPException(401, "invalid or missing X-Model-Secret")


def require_runtime() -> None:
    if RT.store is None or RT.encoders is None:
        raise HTTPException(503, "model-server is still initialising")


# --------------------------------------------------------------------------- #
# image loading (path-confined)
# --------------------------------------------------------------------------- #


def _img_bytes(img: ImagePayload) -> bytes:
    if img.b64:
        if len(img.b64) > MAX_BODY_BYTES:
            raise HTTPException(413, "image payload too large")
        payload = img.b64.split(",", 1)[-1]
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(400, f"invalid base64 image: {exc}") from exc
    if not img.path:
        raise HTTPException(400, "image payload needs 'path' or 'b64'")
    return _read_asset(img.path)


def _read_asset(path: str) -> bytes:
    """Read a file, refusing anything outside ASSET_ROOT.

    ``path`` arrives over HTTP, so without this check
    ``{"path": "../../../etc/shadow"}`` is a file-read primitive.
    """
    root = os.path.realpath(RT.settings.asset_root)
    candidate = path if os.path.isabs(path) else os.path.join(root, path)
    resolved = os.path.realpath(candidate)
    if resolved != root and not resolved.startswith(root + os.sep):
        raise HTTPException(400, "image path escapes ASSET_ROOT")
    if not os.path.isfile(resolved):
        raise HTTPException(404, "image not found")
    if os.path.getsize(resolved) > MAX_BODY_BYTES:
        raise HTTPException(413, "image too large")
    try:
        with open(resolved, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise HTTPException(404, f"cannot read image: {exc}") from exc


# --------------------------------------------------------------------------- #
# platform
# --------------------------------------------------------------------------- #


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    ready = RT.store is not None and RT.encoders is not None
    degraded = RT.degraded() if ready else None
    return {
        "ok": ready and degraded is None,
        "index": RT.store.counts() if RT.store else None,
        "projection": RT.projection.info() if RT.projection else None,
        "degraded": degraded,
    }


@app.get("/v1/models", dependencies=[Depends(require_secret)])
def models():
    s = RT.settings
    return {
        "models": RT.encoders.models() if RT.encoders else {},
        "text_dim": RT.encoders.text.dim if RT.encoders else None,
        "image_dim": RT.encoders.image.dim if RT.encoders else None,
        "embed_dim": s.embed_dim,
        "projection": RT.projection.info() if RT.projection else None,
        "degraded": RT.degraded(),
        "generator": {"model": RT.generator.model_name,
                      "version": RT.generator.model_version} if RT.generator else None,
    }


# --------------------------------------------------------------------------- #
# chunk + embed
# --------------------------------------------------------------------------- #


@app.post("/v1/chunk", dependencies=[Depends(require_secret)])
def chunk(req: ChunkReq):
    if req.overlap_tokens >= req.max_tokens:
        raise HTTPException(400, "overlap_tokens must be < max_tokens")
    chunks = chunk_text(req.text, kind=req.kind, max_tokens=req.max_tokens,
                        overlap_tokens=req.overlap_tokens, tokenizer=RT.tokenizer)
    return {"chunks": [c.to_dict() for c in chunks]}


@app.post("/v1/embed/text", dependencies=[Depends(require_secret)])
def embed_text(req: EmbedTextReq):
    require_runtime()
    try:
        vecs = RT.encode_text(req.texts)
    except Exception as exc:  # noqa: BLE001 - surface, never silently hash
        log.exception("text embedding failed")
        raise HTTPException(502, f"text embedding failed: {exc}") from exc
    return {"model": RT.encoders.text.model_name,
            "model_version": RT.encoders.text.model_version,
            "dim": int(vecs.shape[1]), "vectors": vecs.tolist()}


@app.post("/v1/embed/image", dependencies=[Depends(require_secret)])
def embed_image(req: EmbedImageReq):
    require_runtime()
    try:
        vecs = RT.encode_image([_img_bytes(i) for i in req.images])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("image embedding failed")
        raise HTTPException(502, f"image embedding failed: {exc}") from exc
    return {"model": RT.encoders.image.model_name,
            "model_version": RT.encoders.image.model_version,
            "dim": int(vecs.shape[1]), "vectors": vecs.tolist()}


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #


@app.post("/v1/index/upsert", dependencies=[Depends(require_secret)])
def index_upsert(req: UpsertReq):
    require_runtime()
    try:
        n = RT.store.upsert([e.model_dump() for e in req.entries])
    except ValueError as exc:
        # Rejected before any write, so the index is never half-updated.
        raise HTTPException(400, str(exc)) from exc
    return {"upserted": n, "dirty": RT.store.dirty}


@app.post("/v1/index/remove", dependencies=[Depends(require_secret)])
def index_remove(req: RemoveReq):
    require_runtime()
    return {"removed": RT.store.remove(req.faiss_ids, req.item_ids)}


@app.post("/v1/index/save", dependencies=[Depends(require_secret)])
def index_save():
    require_runtime()
    return {"saved": RT.store.save()}


@app.post("/v1/index/load", dependencies=[Depends(require_secret)])
def index_load():
    require_runtime()
    RT.store.load()
    return {"index": RT.store.counts()}


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


@app.post("/v1/search", dependencies=[Depends(require_secret)])
def search(req: SearchReq):
    require_runtime()
    if not req.query_text and not req.query_image:
        raise HTTPException(400, "provide query_text and/or query_image")
    tvec = RT.encode_text([req.query_text])[0] if req.query_text else None
    ivec = (RT.encode_image([_img_bytes(req.query_image)])[0]
            if req.query_image else None)
    hits = RT.store.search(
        text_vector=tvec, image_vector=ivec, top_k=req.top_k,
        item_ids=req.item_ids, modality=req.modality, fused=req.fused,
        text_weight=RT.settings.text_index_weight,
        image_weight=RT.settings.image_index_weight)
    return {"hits": hits, "models": RT.encoders.models(),
            "degraded": RT.degraded()}


# --------------------------------------------------------------------------- #
# grounded composition
# --------------------------------------------------------------------------- #


def _abstractive_generator(question: str, ctxs: List[dict]) -> str:
    """Draft an answer from the retrieved evidence, then let the citation
    check judge it. Falls back to echoing the question when T5 is unavailable."""
    joined = " ".join(c.get("text", "") for c in ctxs if c.get("text"))[:800]
    spec = {"title": question, "category": "answer", "features": [joined]}
    result = RT.generator.generate(spec, beam_width=4, num_return_sequences=1,
                                   max_len=160)
    drafts = result.get("drafts") or []
    return drafts[0]["text"] if drafts else question


@app.post("/v1/answer/compose", dependencies=[Depends(require_secret)])
def answer_compose(req: ComposeReq):
    require_runtime()
    s = RT.settings

    def embedder(texts: List[str]) -> np.ndarray:
        # A failure here must surface as a 502. Silently returning 0 would make
        # every answer fail its citation check with no explanation.
        try:
            return RT.encode_text(list(texts))
        except Exception as exc:  # noqa: BLE001
            log.exception("composer embedding failed")
            raise HTTPException(502, f"composer embedding failed: {exc}") from exc

    composer = AnswerComposer(s.citation_support_threshold, embedder=embedder)
    contexts = [c.model_dump() for c in req.contexts]
    generator = _abstractive_generator if req.mode == "abstractive" else None
    return composer.compose(req.question, contexts, req.mode, generator)


# --------------------------------------------------------------------------- #
# generation (Project 2)
# --------------------------------------------------------------------------- #


@app.post("/v1/generate/description", dependencies=[Depends(require_secret)])
def generate_description(req: GenerateReq):
    require_runtime()
    try:
        return RT.generator.generate(req.spec, beam_width=req.beam_width,
                                     num_return_sequences=req.num_return_sequences,
                                     max_len=req.max_len)
    except Exception as exc:  # noqa: BLE001
        log.exception("description generation failed")
        raise HTTPException(502, f"description generation failed: {exc}") from exc
