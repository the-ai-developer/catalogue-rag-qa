"""Tests: beam-search draft scoring and the grounded composer's failure modes."""

import numpy as np
import pytest

from app.answer_composer import NOT_GROUNDED, AnswerComposer
from app.generator import sequence_logprobs


# --------------------------------------------------------------------------- #
# draft scoring
# --------------------------------------------------------------------------- #

def test_each_draft_is_scored_from_its_own_beam():
    """Regression: `logprobs[0, step, tok]` scored every draft with beam 0's
    numbers, so the editor's "ranked drafts" all carried rank 1's score.

    scores is [batch=2, steps=2, vocab=3]. Beam 0 always picks token 0 then
    token 1; beam 1 always picks token 2 with p=0.5.
    """
    scores = np.array([
        [[0.9, 0.05, 0.05],     # step 0, beam 0 -> token 0 at p=0.9
         [0.05, 0.9, 0.05]],    # step 1, beam 0 -> token 1 at p=0.9
        [[0.2, 0.3, 0.5],       # step 0, beam 1 -> token 2 at p=0.5
         [0.2, 0.3, 0.5]],      # step 1, beam 1 -> token 2 at p=0.5
    ], dtype=np.float64)
    sequences = [[0, 1], [2, 2]]

    means = sequence_logprobs(sequences, scores, pad_id=None, eos_id=None)
    assert means[0] > means[1], means
    assert means[0] == pytest.approx(np.log(0.9), abs=1e-9)
    assert means[1] == pytest.approx(np.log(0.5), abs=1e-9)


def test_pad_and_eos_are_excluded_from_the_mean():
    scores = np.full((1, 4, 5), 0.2)
    # tokens: real, real, eos, pad
    means = sequence_logprobs([[1, 2, 3, 4]], scores, pad_id=4, eos_id=3)
    assert means[0] == pytest.approx(np.log(0.2), abs=1e-9)


def test_trailing_pad_after_eos_does_not_penalise():
    scores = np.full((1, 4, 5), 0.2)
    long_pad = sequence_logprobs([[1, 2, 3, 4]], scores, pad_id=4, eos_id=3)
    short = sequence_logprobs([[1, 2]], scores, pad_id=4, eos_id=3)
    assert long_pad == pytest.approx(short)


def test_empty_sequence_does_not_divide_by_zero():
    assert sequence_logprobs([[]], np.full((1, 2, 3), 0.5), None, None) == [0.0]


def test_wrong_ranked_scores_raise():
    """A 2-D scores array would silently index the wrong axis before."""
    with pytest.raises(ValueError, match="batch, steps, vocab"):
        sequence_logprobs([[0]], np.full((2, 3), 0.5), None, None)


# --------------------------------------------------------------------------- #
# composer
# --------------------------------------------------------------------------- #

def _ctxs():
    return [
        {"item_id": "a", "sku": "KX-1", "modality": "text", "score": 0.9,
         "text": "The Kestrel bottle holds 12 oz and is dishwasher safe."},
        {"item_id": "a", "sku": "KX-1", "modality": "image", "score": 0.7,
         "text": ""},
    ]


def test_embedder_failure_surfaces_instead_of_scoring_zero():
    """The silent-`except: return 0.0` demoted every sentence to lexical-only
    scoring, so answers failed their citation check with no diagnosable cause.
    The failure must propagate so /v1/answer/compose can answer 502."""

    def broken(texts):
        raise RuntimeError("cuda oom")

    composer = AnswerComposer(0.35, embedder=broken)
    with pytest.raises(RuntimeError, match="cuda oom"):
        composer.compose("Is the Kestrel bottle dishwasher safe?", _ctxs())


def test_embedder_returning_wrong_shape_raises():
    composer = AnswerComposer(0.35, embedder=lambda t: np.zeros((1, 8)))
    with pytest.raises(ValueError, match="expected"):
        composer.compose("Is it dishwasher safe?", _ctxs())


def test_healthy_embedder_keeps_answers_grounded():
    rng = np.random.default_rng(0)

    def embedder(texts):
        # identical text -> identical vector, so cosine is 1 for the question
        out = [l2(rows) for rows in (
            rng.standard_normal(32) for _ in texts)]
        return np.stack(out)

    def l2(v):
        return v / (np.linalg.norm(v) or 1.0)

    composer = AnswerComposer(0.35, embedder=embedder)
    out = composer.compose("Is the Kestrel bottle dishwasher safe?", _ctxs())
    assert out["citation_check"]["passed"] is True
    assert out["answer"] != NOT_GROUNDED


def test_contexts_are_required():
    out = AnswerComposer(0.35).compose("anything?", [])
    assert out["answer"] == NOT_GROUNDED
    assert out["citation_check"]["passed"] is False


def test_unsupported_sentences_are_stripped_and_flagged_not_rendered():
    """Pins the *current* contract for abstractive mode.

    The composer returns the supported sentences plus ``passed=false``, and the
    client (AskPage) refuses to render anything when ``passed`` is false. Note
    the asymmetry with descriptions, where the gate is enforced server-side in
    ``publishDescription`` — a non-browser API consumer of /qa/ask still sees
    the surviving text in the ``answer`` field.
    """
    ctx = [{"item_id": "a", "modality": "text", "score": 0.9,
            "text": "The bottle holds 12 oz and is dishwasher safe."}]
    composer = AnswerComposer(0.35)
    out = composer.compose(
        "What is the capacity?",
        ctx, mode="abstractive",
        generator=lambda q, c: "The bottle holds 12 oz. It is made of solid gold.")
    assert out["citation_check"]["passed"] is False
    assert "gold" not in out["answer"]
    assert "12 oz" in out["answer"]
    stripped = [d for d in out["citation_check"]["details"] if d.get("stripped")]
    assert len(stripped) == 1
    assert stripped[0]["supported"] is False


def test_nothing_supported_returns_the_explicit_refusal():
    ctx = [{"item_id": "a", "modality": "text", "score": 0.2,
            "text": "The bottle is made of stainless steel."}]
    out = AnswerComposer(0.35).compose(
        "What is the warranty on the lunar rover?",
        ctx, mode="abstractive",
        generator=lambda q, c: "The lunar rover carries a 42-month warranty.")
    assert out["answer"] == NOT_GROUNDED
    assert out["citation_check"]["passed"] is False
    assert not out["sentences"]
