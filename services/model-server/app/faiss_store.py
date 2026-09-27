"""FAISS vector store: two per-modality indexes over one shared space.

* ``text``  — SBERT-side chunk vectors
* ``image`` — CLIP-side product-photo vectors

Both are ``IndexIDMap2(IndexFlatIP)`` (inner product == cosine on the
L2-normalised shared space).  ``faiss_id`` is allocated by the Go API and equals
``embeddings.id``, so upserts are idempotent across re-ingests.  A sidecar JSON
maps ids back to catalogue metadata; FAISS files are disposable — Postgres plus
assets can rebuild everything (see docs/operations.md).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .embed import l2_normalise

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss is required: pip install faiss-cpu") from exc

log = logging.getLogger(__name__)

META_FILE = "meta.json"
_INDEX_FILES = {"text": "text.index", "image": "image.index"}


class FaissStore:
    """Thread-safe dual FAISS index with metadata sidecar.

    The index is persisted to ``index_dir``: :meth:`mark_dirty` plus
    :meth:`save_if_dirty` let a background task flush at most every
    ``min_interval`` seconds instead of on every upsert, so a 20-item seed does
    not write 20 times while a 100k-item rebuild still checkpoints regularly.
    """

    def __init__(self, dim: int, index_dir: str = ""):
        self.dim = dim
        self.index_dir = index_dir
        self._lock = threading.RLock()
        self._index = {m: faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
                       for m in ("text", "image")}
        self._ids: Dict[str, set] = {"text": set(), "image": set()}
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._item_index: Dict[str, set] = {}
        self._dirty = False
        self._last_saved = 0.0

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def mark_dirty(self) -> None:
        """Mark the on-disk copy stale. Call after any successful mutation."""
        with self._lock:
            self._dirty = True

    def save_if_dirty(self, min_interval: float = 30.0, force: bool = False) -> bool:
        """Persist when dirty and ``min_interval`` has elapsed.

        Returns True when a write happened. Never raises: a failed autosave must
        not take down the request that triggered it — the index is rebuildable
        from Postgres, and the failure is logged.
        """
        with self._lock:
            if not self._dirty:
                return False
            now = time.monotonic()
            if not force and (now - self._last_saved) < min_interval:
                return False
            try:
                self._save_locked()
                self._dirty = False
                self._last_saved = time.monotonic()
                return True
            except OSError as exc:
                log.error("faiss autosave failed (index still in memory): %s", exc)
                return False

    @staticmethod
    def _remove_from(index, ids: List[int]) -> None:
        """Remove ids, tolerating faiss python API differences."""
        if not ids:
            return
        arr = np.asarray(ids, dtype=np.int64)
        try:
            index.remove_ids(arr)
        except Exception:
            index.remove_ids(faiss.IDSelectorBatch(arr))

    def _check_entry(self, e: Dict[str, Any]) -> int:
        """Validate one upsert entry; raises ValueError with a useful message."""
        vec = np.asarray(e["vector"], dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {vec.shape}")
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"vector dim {vec.shape[0]} != index dim {self.dim}; "
                "the projection and EMBED_DIM disagree — refusing a partial write")
        if not np.all(np.isfinite(vec)):
            raise ValueError("vector contains NaN or Inf")
        if e.get("modality") not in self._index:
            raise ValueError(f"unknown modality {e.get('modality')!r}")
        return int(e["faiss_id"])

    # ------------------------------- writes --------------------------- #
    def upsert(self, entries: Iterable[Dict[str, Any]]) -> int:
        """Insert or replace vectors.  Entry: faiss_id, item_id, modality,
        chunk_id?, asset_id?, text?, vector.

        Every entry is validated before anything is written, so a bad row in the
        middle of a batch cannot leave the index half-updated.
        """
        rows = list(entries)
        prepared = []
        for e in rows:
            fid = self._check_entry(e)
            vec = l2_normalise(np.asarray(e["vector"], np.float32)[None, :])
            prepared.append((fid, vec, e))

        by_mod: Dict[str, List[tuple]] = {"text": [], "image": []}
        with self._lock:
            for fid, vec, e in prepared:
                by_mod[e["modality"]].append((fid, vec, e))
            n = 0
            for mod, batch in by_mod.items():
                if not batch:
                    continue
                for fid, _, _e in batch:
                    if fid in self._ids[mod]:
                        self._remove_from(self._index[mod], [fid])
                        old = self._meta.get(fid)
                        if old:
                            old_item = str(old.get("item_id", ""))
                            if old_item in self._item_index:
                                self._item_index[old_item].discard(fid)
                                if not self._item_index[old_item]:
                                    del self._item_index[old_item]
                ids = np.asarray([r[0] for r in batch], np.int64)
                mat = np.concatenate([r[1] for r in batch], axis=0)
                self._index[mod].add_with_ids(mat, ids)
                self._ids[mod].update(int(r[0]) for r in batch)
                for fid, _, e in batch:
                    item = str(e["item_id"])
                    self._meta[fid] = {
                        "faiss_id": fid, "item_id": item, "modality": mod,
                        "chunk_id": e.get("chunk_id"), "asset_id": e.get("asset_id"),
                        "text": e.get("text", ""),
                    }
                    self._item_index.setdefault(item, set()).add(fid)
                    n += 1
            self._dirty = True
            return n

    def remove(self, faiss_ids: Optional[List[int]] = None,
               item_ids: Optional[List[str]] = None) -> int:
        with self._lock:
            ids: List[int] = list(faiss_ids or [])
            for item in item_ids or []:
                ids.extend(self._item_index.get(str(item), set()))
            ids = sorted(set(int(i) for i in ids))
            for fid in ids:
                mod = self._meta.get(fid, {}).get("modality")
                if mod and fid in self._ids[mod]:
                    self._remove_from(self._index[mod], [fid])
                    self._ids[mod].discard(fid)
                item = self._meta.pop(fid, {}).get("item_id")
                if item and item in self._item_index:
                    self._item_index[item].discard(fid)
                    if not self._item_index[item]:
                        del self._item_index[item]
            if ids:
                self._dirty = True
            return len(ids)

    # ------------------------------- reads ---------------------------- #
    def _search_one(self, mod: str, query: np.ndarray, k: int,
                    item_ids: Optional[List[str]]):
        index = self._index[mod]
        if index.ntotal == 0:
            return []
        # Over-fetch when filtering by item so the filter still leaves `k` rows.
        fetch = k if not item_ids else min(index.ntotal, max(k * 20, 200))
        scores, ids = index.search(np.asarray(query, np.float32)[None, :], fetch)
        wanted = {str(i) for i in item_ids} if item_ids else None
        hits = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:
                continue
            meta = self._meta.get(int(fid))
            if meta is None:
                continue
            if wanted is not None and meta["item_id"] not in wanted:
                continue
            hits.append({**meta, "score": float(score)})
            if len(hits) >= k:
                break
        return hits

    def search(self, *, text_vector: Optional[np.ndarray] = None,
               image_vector: Optional[np.ndarray] = None, top_k: int = 6,
               item_ids: Optional[List[str]] = None,
               modality: str = "any", fused: bool = True,
               text_weight: float = 0.6, image_weight: float = 0.4) -> List[dict]:
        """Top-k over one or both indexes.

        **Fusion is per item, not per vector.** A text chunk and a product photo
        of the same item are two different ``embeddings`` rows with two different
        ``faiss_id``s, so a naive union over ids can never see them as one match.
        Grouping by ``item_id`` first and combining the best score per modality
        is what makes "matched in both text and photo" actually outrank
        "matched in one". Within a fused hit the *best* per-modality row is kept
        as the representative, and the fused score is reported alongside it.
        """
        with self._lock:
            per_mod: Dict[str, List[dict]] = {}
            if text_vector is not None and modality in ("any", "text"):
                per_mod["text"] = self._search_one("text", text_vector, top_k,
                                                    item_ids)
            if image_vector is not None and modality in ("any", "image"):
                per_mod["image"] = self._search_one("image", image_vector, top_k,
                                                     item_ids)

            if not fused or len(per_mod) < 2:
                hits = [r for rows in per_mod.values() for r in rows]
                hits.sort(key=lambda r: -r["score"])
                return hits[:top_k]

            best: Dict[str, Dict[str, dict]] = {}
            for mod, rows in per_mod.items():
                for r in rows:
                    slot = best.setdefault(str(r["item_id"]), {})
                    if mod not in slot or r["score"] > slot[mod]["score"]:
                        slot[mod] = r

            fused_hits: List[dict] = []
            for item, slot in best.items():
                t = slot["text"]["score"] if "text" in slot else None
                i = slot["image"]["score"] if "image" in slot else None
                score = (text_weight * (t or 0.0)) + (image_weight * (i or 0.0))
                if t is None and i is None:
                    continue
                primary = slot["text"] if "text" in slot else slot["image"]
                fused_hits.append({**primary, "score": float(score),
                                   "matched_modalities": sorted(slot.keys()),
                                   "text_score": t, "image_score": i})
            fused_hits.sort(key=lambda r: -r["score"])
            return fused_hits[:top_k]

    # ---------------------------- persistence ------------------------- #
    def save(self, index_dir: str = "") -> str:
        with self._lock:
            directory = index_dir or self.index_dir
            self._save_locked(directory)
            self._dirty = False
            self._last_saved = time.monotonic()
        return directory

    def _save_locked(self, directory: str = "") -> None:
        directory = directory or self.index_dir
        os.makedirs(directory, exist_ok=True)
        for mod, name in _INDEX_FILES.items():
            # Write to a temp file then rename: a crash mid-write must not leave
            # a truncated index that fails to load on the next boot.
            final = os.path.join(directory, name)
            tmp = final + ".tmp"
            faiss.write_index(self._index[mod], tmp)
            os.replace(tmp, final)
        meta_path = os.path.join(directory, META_FILE)
        tmp = meta_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({str(k): v for k, v in self._meta.items()}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, meta_path)

    def load(self, index_dir: str = "") -> None:
        with self._lock:
            directory = index_dir or self.index_dir
            for mod, name in _INDEX_FILES.items():
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    self._index[mod] = faiss.read_index(path)
            meta_path = os.path.join(directory, META_FILE)
            if os.path.exists(meta_path):
                with open(meta_path) as fh:
                    raw = json.load(fh)
                self._meta = {int(k): v for k, v in raw.items()}
                self._item_index = {}
                self._ids = {"text": set(), "image": set()}
                for fid, meta in self._meta.items():
                    self._item_index.setdefault(meta["item_id"], set()).add(fid)
                    self._ids.get(meta.get("modality", "text"), set()).add(fid)
            self._dirty = False
            self._last_saved = time.monotonic()

    def counts(self) -> dict:
        with self._lock:
            return {"text": self._index["text"].ntotal,
                    "image": self._index["image"].ntotal,
                    "meta": len(self._meta)}
