"""Spec → description generation with beam search (Project 2, contract §2).

Production path: fine-tuned T5 (``DESC_MODEL_DIR``) via ``model.generate`` beam
search.  If no checkpoint or ``transformers`` is installed, a deterministic
template generator keeps the pipeline alive and says so loudly in
``model_version`` so the editor gate and the audit trail both see it.

Scores are mean token log-probabilities **per returned sequence**, so drafts are
comparable across beams — see :func:`sequence_logprobs`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .linearise import linearise_spec

log = logging.getLogger(__name__)

# Only these failures mean "no usable model". Anything else (OOM, a corrupt
# checkpoint, a CUDA error) is a real fault and must not be disguised as an
# absent checkpoint by falling through to template text.
_MISSING_MODEL_ERRORS = (ImportError, ModuleNotFoundError, FileNotFoundError,
                         NotADirectoryError, OSError)


def sequence_logprobs(sequences, step_scores, pad_id: Optional[int],
                      eos_id: Optional[int]) -> List[float]:
    """Mean token log-probability for each returned sequence.

    ``step_scores`` is a per-step tensor of shape ``[batch, steps, vocab]`` (as
    ``transformers`` returns with ``output_scores=True``), where ``batch`` indexes
    the *returned sequences*, not the beams.  Indexing batch 0 for every sequence
    — the obvious bug — scores all drafts with the first beam's numbers, which
    makes the editor's "ranked drafts" ranking meaningless.

    Padding after the end-of-sequence token is excluded, as is the eos token
    itself, so a longer beam is not penalised for trailing pad.
    """
    if step_scores.ndim != 3:
        raise ValueError(
            f"expected step scores shaped (batch, steps, vocab), got {step_scores.shape}")
    logprobs = np.log(np.maximum(step_scores, 1e-30))
    skip = {int(t) for t in (pad_id, eos_id) if t is not None}
    out: List[float] = []
    for i, seq in enumerate(sequences):
        tokens = list(seq)
        total, n = 0.0, 0
        for step, tok in enumerate(tokens):
            if step >= logprobs.shape[1]:
                break
            if int(tok) in skip:
                continue
            total += float(logprobs[i, step, int(tok)])
            n += 1
        out.append(total / max(1, n))
    return out


class DescriptionGenerator:
    """Beam-search spec→text generator with a safe template fallback."""

    def __init__(self, desc_model_dir: str = "", device: str = "cpu",
                 seed: Optional[int] = None):
        self.desc_model_dir = desc_model_dir
        self.device = device
        self.seed = seed
        self._model = None
        self._tok = None
        self._torch = None
        self.model_name = "t5-small-desc"
        self.model_version = "uninitialised"

    # ------------------------------------------------------------------ #
    def _load(self) -> bool:
        """Return True when a real seq2seq model is ready.

        Raises on a fault that is *not* "there is no checkpoint here", so a CUDA
        OOM during generation surfaces as a failed job instead of a silently
        templated description.
        """
        if self._model is not None:
            return True
        try:
            import torch  # lazy
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy
        except ImportError as exc:
            self.model_version = "template-fallback:no-transformers"
            log.warning("transformers unavailable (%s); template fallback in use", exc)
            return False

        source = (self.desc_model_dir
                  if self.desc_model_dir and os.path.isdir(self.desc_model_dir)
                  else "t5-small")
        try:
            self._tok = AutoTokenizer.from_pretrained(source)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(source).to(
                self.device).eval()
        except _MISSING_MODEL_ERRORS as exc:
            if source != "t5-small" or not isinstance(exc, (OSError,)):
                # A configured checkpoint that will not load is an operator error.
                self.model_version = f"load-failed:{source}"
                raise RuntimeError(
                    f"description checkpoint {source!r} could not be loaded: {exc}") from exc
            self.model_version = "template-fallback:no-checkpoint"
            log.warning("no fine-tuned checkpoint at %s (%s); using base t5-small",
                        self.desc_model_dir, exc)
            return False
        except Exception as exc:  # noqa: BLE001 - corrupt weights, bad config…
            self.model_version = f"load-failed:{source}"
            raise RuntimeError(
                f"description checkpoint {source!r} could not be loaded: {exc}") from exc

        fine_tuned = source == self.desc_model_dir
        self.model_version = f"{source}:{'fine-tuned' if fine_tuned else 'base-not-fine-tuned'}"
        self._torch = torch
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def _template_drafts(spec: Dict, num: int) -> List[dict]:
        """Deterministic, spec-faithful drafts so the pipeline works anywhere."""
        name = spec.get("title") or f"{spec.get('category', 'item')}"
        material = spec.get("material") or ""
        feats: Sequence[str] = spec.get("features") or []
        dims = spec.get("dimensions") or {}
        dim_text = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(dims.items()))
        variants = [
            "{name}. {mat_clause}{dim_clause} {feat_clause}.",
            "Meet the {name}: {feat_clause_lower}{mat_clause}{dim_clause}.",
            "{name} — {feat_clause_lower}{dim_clause}{mat_clause}.",
        ]
        drafts = []
        for i in range(min(num, len(variants))):
            text = variants[i].format(
                name=name,
                mat_clause=f"Made from {material}. " if material else "",
                dim_clause=f"Dimensions: {dim_text}. " if dim_text else "",
                feat_clause="Features " + ", ".join(feats) if feats else
                "Built for everyday use",
                feat_clause_lower=("featuring " + ", ".join(feats) + ". ") if feats
                else "built for everyday use. ",
            )
            drafts.append({"rank": i + 1, "text": " ".join(text.split()),
                           "score": round(-0.4 - 0.1 * i, 4)})
        return drafts

    # ------------------------------------------------------------------ #
    def generate(self, spec: Dict, *, beam_width: int = 4,
                 num_return_sequences: int = 3, max_len: int = 192) -> dict:
        """Return ranked drafts ``[{rank, text, score}]`` + provenance."""
        num = max(1, min(num_return_sequences, beam_width))
        if not self._load():
            return {"model": self.model_name, "model_version": self.model_version,
                    "drafts": self._template_drafts(spec, num), "latency_ms": 0}

        if self.seed is not None:
            self._torch.manual_seed(self.seed)
        prompt = linearise_spec(spec)
        t0 = time.monotonic()
        inputs = self._tok(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                num_beams=max(beam_width, num),
                num_return_sequences=num,
                max_new_tokens=max_len,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # [batch, steps, vocab] where batch indexes the returned sequences.
        step_scores = self._torch.stack(out.scores, dim=1).detach().cpu().numpy()
        means = sequence_logprobs(out.sequences, step_scores,
                                  self._tok.pad_token_id, self._tok.eos_token_id)
        drafts = []
        for i, seq in enumerate(out.sequences):
            text = self._tok.decode(seq, skip_special_tokens=True).strip()
            drafts.append({"rank": i + 1, "text": text,
                           "score": round(means[i], 4)})
        latency = int((time.monotonic() - t0) * 1000)
        return {"model": self.model_name, "model_version": self.model_version,
                "drafts": drafts, "latency_ms": latency}
