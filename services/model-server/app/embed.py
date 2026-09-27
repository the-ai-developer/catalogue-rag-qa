"""Dual embedding paths — SBERT text + CLIP image — both L2-normalised.

Heavy dependencies (torch / transformers / sentence-transformers) are imported
lazily so tests can run with the deterministic :class:`HashEmbedder` on CPU
without downloading weights.  Every encoder exposes ``model_name``,
``model_version`` and ``dim`` for provenance in API responses.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from typing import List, Sequence, Union

import numpy as np

log = logging.getLogger(__name__)

ImageInput = Union[str, bytes]  # filesystem path or raw bytes


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero vectors are left untouched."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class HashEmbedder:
    """Deterministic pseudo-embedder for tests and CPU smoke runs.

    Vectors come from SHA-256 of the input seeded into a fixed RNG — identical
    input always yields the identical unit vector, unrelated inputs are nearly
    orthogonal, which is enough to exercise retrieval plumbing.
    """

    def __init__(self, dim: int = 512, model_name: str = "hash-embedder"):
        self.dim = dim
        self.model_name = model_name
        self.model_version = "hash-1"

    def _one(self, key: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def encode(self, items: Sequence[Union[str, ImageInput]]) -> np.ndarray:
        out = []
        for it in items:
            key = it if isinstance(it, bytes) else str(it).encode()
            out.append(self._one(key))
        return l2_normalise(np.stack(out)) if out else np.zeros((0, self.dim), np.float32)


class TextEmbedder:
    """SBERT text encoder (sentence-transformers), L2-normalised output."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model_version = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        return getattr(self, "_dim", 0)

    @dim.setter
    def dim(self, value: int) -> None:
        self._dim = value

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(list(texts), convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        return l2_normalise(np.asarray(vecs, dtype=np.float32))


class ImageEmbedder:
    """CLIP image-tower encoder, L2-normalised output."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model_version = model_name
        self._model = None
        self._dim = 0

    def _load(self):
        if self._model is None:
            import torch  # lazy
            from transformers import CLIPModel, CLIPProcessor  # lazy

            self._model = (
                CLIPModel.from_pretrained(self.model_name).to(self.device).eval(),
                CLIPProcessor.from_pretrained(self.model_name),
                torch,
            )
            self._dim = int(self._model[0].config.projection_dim)
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def _read(self, img: ImageInput):
        from PIL import Image  # lazy

        if isinstance(img, bytes):
            return Image.open(io.BytesIO(img)).convert("RGB")
        if img.startswith("data:"):
            return Image.open(io.BytesIO(base64.b64decode(img.split(",", 1)[1]))).convert("RGB")
        return Image.open(img).convert("RGB")

    def encode(self, images: Sequence[ImageInput]) -> np.ndarray:
        model, processor, torch = self._load()
        pils = [self._read(img) for img in images]
        batch = processor(images=pils, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = model.get_image_features(**batch)
        return l2_normalise(feats.detach().cpu().numpy().astype(np.float32))


class EncoderBundle:
    """The two production encoders (or hash fallbacks) plus provenance."""

    def __init__(self, text, image):
        self.text = text
        self.image = image

    def models(self) -> dict:
        return {
            "text": f"{self.text.model_name}@{self.text.model_version}",
            "image": f"{self.image.model_name}@{self.image.model_version}",
        }


def build_encoders(settings, *, force_hash: bool = False) -> EncoderBundle:
    """Construct production encoders, honouring ``EMBEDDER_BACKEND=hash``."""
    import os

    backend = os.environ.get("EMBEDDER_BACKEND", "")
    if force_hash or backend == "hash":
        return EncoderBundle(HashEmbedder(settings.embed_dim, "hash-text"),
                             HashEmbedder(settings.embed_dim, "hash-image"))
    try:
        return EncoderBundle(
            TextEmbedder(settings.text_model_name, settings.model_device),
            ImageEmbedder(settings.image_model_name, settings.model_device),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        log.warning("falling back to hash embedders: %s", exc)
        return EncoderBundle(HashEmbedder(settings.embed_dim, "hash-text"),
                             HashEmbedder(settings.embed_dim, "hash-image"))
