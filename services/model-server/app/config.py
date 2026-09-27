"""Environment-driven configuration.

Variable names match ``catalogue-ai/.env.example`` exactly.  Every value is
read at call time so tests can monkeypatch ``os.environ`` safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env(name: str, default: str) -> str:
    """Return the environment variable ``name`` or ``default``."""
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    """Return the environment variable ``name`` parsed as float, or ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    """Return the environment variable ``name`` parsed as int, or ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    """Resolved model-server settings (see ``.env.example``)."""

    model_device: str
    text_model_name: str
    image_model_name: str
    desc_model_dir: str
    projection_path: str
    embed_dim: int
    faiss_index_dir: str
    asset_root: str
    text_index_weight: float
    image_index_weight: float
    citation_support_threshold: float
    max_chunk_tokens: int
    chunk_overlap_tokens: int
    model_seed: Optional[int]
    shared_secret: str
    faiss_autosave_secs: float


def get_settings() -> Settings:
    """Build settings from the current environment."""
    seed_raw = os.environ.get("MODEL_SEED", "")
    seed: Optional[int] = int(seed_raw) if seed_raw not in ("", "none", "None") else None
    return Settings(
        model_device=_env("MODEL_DEVICE", "cpu"),
        text_model_name=_env("TEXT_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"),
        image_model_name=_env("IMAGE_MODEL_NAME", "openai/clip-vit-base-patch32"),
        desc_model_dir=_env("DESC_MODEL_DIR", "ml/checkpoints/t5-description"),
        projection_path=_env("PROJECTION_PATH", "ml/checkpoints/projection/projection.pt"),
        embed_dim=_env_int("EMBED_DIM", 512),
        faiss_index_dir=_env("FAISS_INDEX_DIR", "/data/faiss"),
        asset_root=_env("ASSET_ROOT", "/data/assets"),
        text_index_weight=_env_float("TEXT_INDEX_WEIGHT", 0.6),
        image_index_weight=_env_float("IMAGE_INDEX_WEIGHT", 0.4),
        citation_support_threshold=_env_float("CITATION_SUPPORT_THRESHOLD", 0.35),
        max_chunk_tokens=_env_int("MAX_CHUNK_TOKENS", 220),
        chunk_overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 40),
        model_seed=seed,
        shared_secret=os.environ.get("MODEL_SHARED_SECRET", "").strip(),
        faiss_autosave_secs=_env_float("FAISS_AUTOSAVE_SECS", 30.0),
    )
