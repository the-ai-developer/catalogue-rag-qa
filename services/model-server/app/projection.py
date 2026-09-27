"""Shared-space projection (contract: "Shared-space mapping").

SBERT and CLIP do not share an output dimension — ``all-mpnet-base-v2`` emits
768 dims, ``clip-vit-base-patch32``'s image tower emits 512.  One matrix cannot
map both, so :class:`ProjectionMapper` holds **two** bias-free linear maps
(``text`` and ``image``) into the same ``embed_dim``-wide space, each followed
by L2 re-normalisation.  After mapping, a single inner product compares a text
chunk with a product photo.

Weights are trained with symmetric InfoNCE on item-level (text, image) pairs by
``ml/training/train_projection.py``.

On-disk format is an ``.npz`` with ``text`` and ``image`` keys.  A legacy
single-matrix file (key ``weight``) is still readable and is applied to **both**
towers, which is only correct when the two encoders happen to share a dimension
— :attr:`legacy` records that so ``/v1/models`` can say so.

With no weights at all the mapper is the identity (pad/truncate to
``embed_dim``).  That is *not* a shared space: the two towers are not aligned
and cross-modal recall is noise.  Callers must treat ``trained is False`` as
degraded rather than silently serving meaningless scores, so
:attr:`trained` is surfaced on ``/v1/models``, in ``/readyz`` and in every
``/v1/search`` response.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np

from .embed import l2_normalise

log = logging.getLogger(__name__)

TEXT_KEY = "text"
IMAGE_KEY = "image"
LEGACY_KEY = "weight"


class ProjectionMapper:
    """Two-headed bias-free linear map + re-normalise into one shared space."""

    def __init__(self, embed_dim: int, weight: Optional[np.ndarray] = None,
                 text_weight: Optional[np.ndarray] = None,
                 image_weight: Optional[np.ndarray] = None,
                 path: str = "", trained: bool = False, legacy: bool = False):
        self.embed_dim = embed_dim
        self.path = path
        self.trained = trained
        self.legacy = legacy
        # (out_dim, in_dim) matrices; transposed at call time.
        self.text_weight = _as_matrix(text_weight)
        self.image_weight = _as_matrix(image_weight)
        if self.text_weight is None and self.image_weight is None and weight is not None:
            self.text_weight = _as_matrix(weight)
            self.image_weight = self.text_weight
            self.legacy = True

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str, embed_dim: int) -> "ProjectionMapper":
        """Load ``projection.npz`` (preferred) or ``projection.pt`` (torch).

        Torch checkpoints are converted to numpy immediately so serving needs no
        torch.  Anything unreadable logs an error and yields an untrained
        mapper — never a silently wrong space.
        """
        npz = path.rsplit(".", 1)[0] + ".npz"
        if os.path.exists(npz):
            try:
                data = np.load(npz)
                keys = set(data.files)
                if TEXT_KEY in keys and IMAGE_KEY in keys:
                    return cls(embed_dim, text_weight=data[TEXT_KEY],
                               image_weight=data[IMAGE_KEY], path=path,
                               trained=True)
                if LEGACY_KEY in keys:
                    log.warning("projection %s uses the deprecated single-matrix "
                                "format; retrain with train_projection.py for a "
                                "true two-tower shared space", npz)
                    return cls(embed_dim, weight=data[LEGACY_KEY], path=path,
                               trained=True, legacy=True)
                log.error("projection %s has no usable keys (found %s)",
                          npz, sorted(keys))
            except (OSError, ValueError) as exc:
                log.error("could not read projection %s: %s", npz, exc)
        elif os.path.exists(path):
            try:
                import torch  # lazy

                state = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and TEXT_KEY in state and IMAGE_KEY in state:
                    return cls(embed_dim,
                               text_weight=state[TEXT_KEY].detach().cpu().numpy(),
                               image_weight=state[IMAGE_KEY].detach().cpu().numpy(),
                               path=path, trained=True)
                weight = state[LEGACY_KEY] if isinstance(state, dict) else state
                log.warning("projection %s uses the deprecated single-matrix "
                            "format; retrain with train_projection.py", path)
                return cls(embed_dim, weight=weight.detach().cpu().numpy(),
                           path=path, trained=True, legacy=True)
            except Exception as exc:  # noqa: BLE001 - torch raises many types
                log.error("could not read projection %s: %s", path, exc)
        else:
            log.error("projection weights missing (%s); the two towers are NOT "
                      "aligned, so cross-modal retrieval is meaningless — run "
                      "ml/training/train_projection.py", path)
        return cls(embed_dim, path=path, trained=False)

    def save(self, path: str) -> None:
        """Persist as ``.npz`` with ``text``/``image`` keys (plus ``.pt``)."""
        if self.text_weight is None or self.image_weight is None:
            raise ValueError("refusing to save an untrained projection")
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        payload: Dict[str, np.ndarray] = {
            TEXT_KEY: self.text_weight,
            IMAGE_KEY: self.image_weight,
        }
        np.savez(path.rsplit(".", 1)[0] + ".npz", **payload)
        try:
            import torch  # lazy

            torch.save({k: torch.from_numpy(v) for k, v in payload.items()}, path)
        except Exception as exc:  # noqa: BLE001 - torch is optional at save time
            log.warning("torch save failed (%s); the .npz copy is authoritative", exc)

    # ------------------------------------------------------------------ #
    def _adapt(self, x: np.ndarray, weight: Optional[np.ndarray]) -> np.ndarray:
        """Pad/truncate the last axis to the matrix input dim (or embed_dim)."""
        target = weight.shape[1] if weight is not None else self.embed_dim
        if x.shape[-1] == target:
            return x
        out = np.zeros(x.shape[:-1] + (target,), np.float32)
        n = min(x.shape[-1], target)
        out[..., :n] = x[..., :n]
        return out

    def _apply(self, vectors: np.ndarray, which: str) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float32)
        weight = self.text_weight if which == TEXT_KEY else self.image_weight
        x = self._adapt(x, weight)
        if weight is not None:
            x = x @ weight.T
        return l2_normalise(x)

    def project_text(self, vectors: np.ndarray) -> np.ndarray:
        """Map text-tower vectors into the shared space."""
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            return self.project_text(x[None, :])[0]
        return self._apply(x, TEXT_KEY)

    def project_image(self, vectors: np.ndarray) -> np.ndarray:
        """Map image-tower vectors into the shared space."""
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            return self.project_image(x[None, :])[0]
        return self._apply(x, IMAGE_KEY)

    def __call__(self, vectors: np.ndarray, which: str = TEXT_KEY) -> np.ndarray:
        return self._apply(np.asarray(vectors, dtype=np.float32), which)

    def info(self) -> dict:
        return {
            "trained": self.trained,
            "legacy_single_matrix": self.legacy,
            "embed_dim": self.embed_dim,
            "text_weight_shape": (None if self.text_weight is None
                                  else list(self.text_weight.shape)),
            "image_weight_shape": (None if self.image_weight is None
                                   else list(self.image_weight.shape)),
            "path": self.path,
        }


def _as_matrix(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return arr if arr.ndim == 2 else None
