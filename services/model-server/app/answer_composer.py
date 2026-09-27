"""Grounded answer composition + citation check (contract: /v1/answer/compose).

Product rule: **grounded or nothing**.  Answers are composed strictly from the
retrieved contexts; every emitted sentence carries citations and a support
score.  Sentences that cannot be supported by retrieved evidence are stripped
and reported, and if nothing supported remains the answer is an explicit
"no grounded evidence" message with ``citation_check.passed = false``.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

NOT_GROUNDED = "I don't have grounded catalogue evidence for that."

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VISUAL_CUES = {"look", "looks", "looking", "image", "images", "photo", "photos",
                "picture", "colour", "colours", "color", "colors", "appearance",
                "show", "see", "visual"}
# Function words are ignored for relevance so "is the … on the …" questions
# cannot match by stopwords alone.  Negations (not/no) are deliberately kept:
# "not dishwasher safe" must not score like "dishwasher safe".
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "done", "of", "in", "on", "at", "to", "for",
    "with", "what", "which", "who", "whom", "whose", "how", "when", "where",
    "why", "it", "its", "this", "that", "these", "those", "and", "or", "but",
    "if", "then", "than", "so", "as", "up", "out", "about", "into", "over",
    "after", "before", "between", "from", "by", "off", "we", "you", "i", "he",
    "she", "they", "them", "his", "her", "their", "our", "your", "my", "me",
    "us", "can", "will", "just", "should", "now", "also", "have", "has", "had",
    "would", "could", "may", "might", "must", "shall", "let", "there", "here",
}


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _token_f1(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(ta), overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


class AnswerComposer:
    """Composes answers from retrieved contexts with per-sentence citations."""

    def __init__(self, support_threshold: float = 0.35,
                 embedder: Optional[Callable[[Sequence[str]], object]] = None):
        self.support_threshold = support_threshold
        self._embed = embedder  # optional: texts -> L2-normalised vectors

    # ------------------------------------------------------------------ #
    def _embed_cos(self, a: str, b: str) -> float:
        """Cosine similarity from the shared space.

        Deliberately does *not* swallow embedder errors. Returning 0.0 on
        failure would silently demote every answer to lexical-only scoring and
        then fail the citation check with no diagnosable cause — the worst
        possible failure mode for a "grounded or nothing" product. The caller
        (/v1/answer/compose) turns an exception here into a 502.
        """
        if self._embed is None or not b.strip():
            return 0.0
        vecs = np.asarray(self._embed([a, b]), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] < 2:
            raise ValueError(
                f"embedder returned shape {vecs.shape}; expected (2, dim)")
        return float(np.dot(vecs[0], vecs[1]))

    def _support(self, sentence: str, context_text: str) -> float:
        """Blend of lexical overlap and (optional) embedding similarity."""
        score = _token_f1(sentence, context_text)
        if self._embed is not None and context_text.strip():
            score = 0.6 * score + 0.4 * self._embed_cos(sentence, context_text)
        return round(min(1.0, score), 4)

    def _best_support(self, sentence: str, contexts: Sequence[dict]):
        best, idx = 0.0, -1
        for i, ctx in enumerate(contexts):
            if ctx.get("modality") == "image":
                continue
            s = self._support(sentence, ctx.get("text", ""))
            if s > best:
                best, idx = s, i
        return best, idx

    # ----------------------------- extractive ------------------------- #
    def _extractive(self, question: str, contexts: Sequence[dict]) -> dict:
        """Select the most question-relevant evidence sentences verbatim."""
        candidates = []  # (rel, ctx_idx, sentence, order)
        for ci, ctx in enumerate(contexts):
            if ctx.get("modality") == "image":
                continue
            for si, sent in enumerate(_sentences(ctx.get("text", ""))):
                lex = _token_f1(question, sent)
                cos = self._embed_cos(question, sent)
                rel = 0.5 * lex + 0.5 * cos if self._embed is not None else lex
                # a candidate must be lexically or strongly semantically tied
                # to the question — never relevance by accident
                if rel >= 0.15 and (lex >= 0.05 or cos >= 0.6):
                    candidates.append((round(rel, 4), ci, sent, (ci, si)))
        candidates.sort(key=lambda c: (-c[0], c[3]))
        chosen = candidates[:4]

        sentences: List[dict] = []
        if chosen:
            chosen.sort(key=lambda c: c[3])  # readable order
            for out_i, (rel, ci, sent, _) in enumerate(chosen):
                sentences.append({
                    "index": out_i, "text": sent,
                    "citations": [{"context_index": ci, "score": round(rel, 4)}],
                })
        # honest visual-evidence sentence when the question is about looks
        if _tokens(question) & _VISUAL_CUES:
            for ci, ctx in enumerate(contexts):
                if ctx.get("modality") == "image":
                    sentences.append({
                        "index": len(sentences),
                        "text": f"Product photograph evidence was retrieved for item "
                                f"{ctx.get('sku') or ctx.get('item_id', '')}.".strip(),
                        "citations": [{"context_index": ci, "score": round(float(ctx.get("score", 0.5)), 4)}],
                    })
        return self._finalise(question, contexts, sentences, "extractive")

    # ----------------------------- abstractive ------------------------ #
    def _abstractive(self, question: str, contexts: Sequence[dict],
                     generator: Callable[[str, Sequence[dict]], str]) -> dict:
        raw = generator(question, contexts)
        draft = [{"index": i, "text": s, "citations": []}
                 for i, s in enumerate(_sentences(raw))]
        sentences, stripped = [], []
        for sent in draft:
            support, ci = self._best_support(sent["text"], contexts)
            if support >= self.support_threshold and ci >= 0:
                sentences.append({**sent, "index": len(sentences),
                                  "citations": [{"context_index": ci, "score": support}]})
            else:
                stripped.append({**sent, "support_score": support})
        result = self._finalise(question, contexts, sentences, "abstractive")
        for s in stripped:
            result["citation_check"]["details"].append({
                "sentence_index": s["index"], "supported": False,
                "support_score": s["support_score"], "stripped": True,
            })
        if stripped:
            result["citation_check"]["passed"] = False
        return result

    # ----------------------------- shared ----------------------------- #
    def _finalise(self, question: str, contexts: Sequence[dict],
                  sentences: List[dict], mode: str) -> dict:
        details = []
        for sent in sentences:
            support, _ = self._best_support(sent["text"], contexts)
            # a sentence whose citations point at image evidence is supported by
            # that evidence: use the retrieval score as its support floor
            for cit in sent.get("citations", []):
                ctx = contexts[cit["context_index"]] if cit["context_index"] < len(contexts) else {}
                if ctx.get("modality") == "image":
                    support = max(support, float(cit.get("score", 0.0)))
            details.append({"sentence_index": sent["index"],
                            "supported": support >= self.support_threshold,
                            "support_score": support})
        passed = bool(sentences) and all(d["supported"] for d in details)
        answer = " ".join(s["text"] for s in sentences).strip()
        if not sentences or not passed:
            answer = NOT_GROUNDED
            passed = False
        return {"answer": answer, "mode": mode, "sentences": sentences,
                "citation_check": {"passed": passed, "details": details}}

    # ------------------------------------------------------------------ #
    def compose(self, question: str, contexts: Sequence[dict],
                mode: str = "extractive",
                generator: Optional[Callable[[str, Sequence[dict]], str]] = None) -> dict:
        """Compose a grounded answer.

        ``contexts``: [{item_id, sku?, chunk_id?, asset_id?, modality, score,
        text}] — the top-k retrieval result.  Abstractive mode needs
        ``generator(question, contexts) -> str`` and always runs the citation
        check afterwards.
        """
        if not contexts:
            return {"answer": NOT_GROUNDED, "mode": mode, "sentences": [],
                    "citation_check": {"passed": False,
                                       "details": [{"sentence_index": 0,
                                                    "supported": False,
                                                    "support_score": 0.0,
                                                    "note": "no contexts retrieved"}]}}
        if mode == "abstractive" and generator is not None:
            return self._abstractive(question, contexts, generator)
        return self._extractive(question, contexts)
