"""Token-aware text chunking (contract: ``POST /v1/chunk``).

Sentence-aware greedy packing: sentences are packed into windows of at most
``max_tokens`` subword tokens; consecutive windows overlap by roughly
``overlap_tokens`` so evidence spanning a boundary stays retrievable.  Output is
deterministic (stable ordinals) for a given tokenizer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Chunk:
    """One chunk of source text, ready for embedding."""

    ordinal: int
    text: str
    token_count: int

    def to_dict(self) -> dict:
        return {"ordinal": self.ordinal, "text": self.text,
                "token_count": self.token_count}


class Tokenizer:
    """Subword tokenizer with a graceful regex fallback (no downloads needed)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or "whitespace-fallback"
        self._tok = None
        if model_name:
            try:  # lazy: only load HF tokenizer when a real model is configured
                from transformers import AutoTokenizer  # type: ignore

                self._tok = AutoTokenizer.from_pretrained(model_name)
            except Exception as exc:  # noqa: BLE001 - offline / missing weights
                log.warning("tokenizer %s unavailable (%s); counting whitespace "
                            "tokens, so MAX_CHUNK_TOKENS means words", model_name, exc)
                self._tok = None

    @property
    def name(self) -> str:
        return self.model_name if self._tok is not None else "whitespace-fallback"

    def encode(self, text: str) -> List[int]:
        if self._tok is not None:
            return self._tok.encode(text, add_special_tokens=False)
        return list(range(len(text.split())))

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return len(text.split())


def _split_sentences(text: str) -> List[str]:
    """Split on sentence terminators and newlines.

    Called *before* whitespace normalisation: collapsing the text first would
    delete every newline and make the newline branch of the pattern dead.
    """
    parts = [" ".join(p.split()) for p in _SENT_RE.split(text.strip())]
    return [p for p in parts if p]


def chunk_text(text: str, *, kind: str = "copy", max_tokens: int = 220,
               overlap_tokens: int = 40,
               tokenizer: Optional[Tokenizer] = None) -> List[Chunk]:
    """Chunk ``text`` into overlapping token windows of at most ``max_tokens``.

    ``kind`` is carried by the caller into ``chunks.source_kind`` (title, spec,
    copy, feature, dimension) — it does not influence chunking itself.
    Raises ValueError on a nonsensical window so a bad request cannot silently
    produce one giant chunk.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < max_tokens")
    tok = tokenizer or Tokenizer()
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Chunk] = []
    current: List[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            body = " ".join(current)
            chunks.append(Chunk(ordinal=len(chunks), text=body,
                                token_count=tok.count(body)))
            # carry trailing sentences as overlap for the next window
            carried: List[str] = []
            carried_tokens = 0
            for sent in reversed(current):
                t = tok.count(sent)
                if carried_tokens + t > overlap_tokens and carried:
                    break
                carried.insert(0, sent)
                carried_tokens += t
            current = carried
            current_tokens = carried_tokens

    for sent in sentences:
        t = tok.count(sent)
        if t > max_tokens:  # oversized sentence: hard-split on words
            flush()
            buf: List[str] = []
            for w in sent.split():
                buf.append(w)
                if tok.count(" ".join(buf)) >= max_tokens:
                    chunks.append(Chunk(ordinal=len(chunks), text=" ".join(buf),
                                        token_count=tok.count(" ".join(buf))))
                    # keep a tail of words so a fact split across the boundary
                    # stays retrievable
                    buf = buf[-max(1, overlap_tokens // 2):]
            if buf:
                # These are word fragments, not sentences, so they bypass the
                # overlap logic above and start the next window directly.
                current = list(buf)
                current_tokens = tok.count(" ".join(current))
            continue
        if current_tokens + t > max_tokens and current:
            flush()
        current.append(sent)
        current_tokens += t
    flush()

    # re-number ordinals so overlap windows keep stable order
    return [Chunk(ordinal=i, text=c.text, token_count=c.token_count)
            for i, c in enumerate(chunks)]
