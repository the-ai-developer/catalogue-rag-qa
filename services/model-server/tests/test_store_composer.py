"""Tests: FAISS store behaviour + grounded answer composer (CPU, no downloads)."""

import numpy as np
import pytest

from app.answer_composer import NOT_GROUNDED, AnswerComposer
from app.embed import HashEmbedder, l2_normalise
from app.faiss_store import FaissStore


def _vec(seed: int, dim: int = 32) -> list:
    rng = np.random.default_rng(seed)
    return l2_normalise(rng.standard_normal(dim)).tolist()


@pytest.fixture()
def store(tmp_path):
    return FaissStore(dim=32, index_dir=str(tmp_path))


class TestFaissStore:
    def test_upsert_search_remove(self, store):
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "chunk_id": "c1", "text": "dishwasher safe", "vector": _vec(1)},
            {"faiss_id": 2, "item_id": "b", "modality": "image",
             "asset_id": "i1", "text": "", "vector": _vec(2)},
        ])
        assert store.counts() == {"text": 1, "image": 1, "meta": 2}

        hits = store.search(text_vector=np.asarray(_vec(1)), top_k=2)
        assert hits and hits[0]["faiss_id"] == 1
        assert hits[0]["score"] > 0.99  # identical vector -> cosine ~ 1

        # upsert same id replaces
        store.upsert([{"faiss_id": 1, "item_id": "a", "modality": "text",
                       "chunk_id": "c1", "text": "updated", "vector": _vec(7)}])
        assert store.counts()["text"] == 1

        store.remove(item_ids=["a"])
        assert store.counts() == {"text": 0, "image": 1, "meta": 1}

    def test_fusion_combines_text_and_image_for_the_same_item(self, store):
        """Fusion must be per item.

        Item "a" has one text row and one photo row with *different* faiss_ids.
        Item "b" has a text row only. Per-modality top-k is taken with k=1, so a
        naive union over faiss_id sees three unrelated rows and cannot tell that
        "a" matched twice. Grouping by item_id is what makes "a" win.
        """
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "text": "bottle", "vector": _vec(1)},
            {"faiss_id": 3, "item_id": "a", "modality": "image",
             "text": "", "vector": _vec(1)},          # same direction
            {"faiss_id": 2, "item_id": "b", "modality": "text",
             "text": "bag", "vector": _vec(2)},       # weaker text match
        ])
        hits = store.search(text_vector=np.asarray(_vec(1)),
                            image_vector=np.asarray(_vec(1)), top_k=1,
                            fused=True)
        assert [h["item_id"] for h in hits] == ["a"]
        assert hits[0]["matched_modalities"] == ["image", "text"]
        # both sides contributed: 0.6*1.0 + 0.4*1.0
        assert hits[0]["score"] == pytest.approx(1.0, abs=1e-3)
        assert hits[0]["text_score"] == pytest.approx(1.0, abs=1e-3)
        assert hits[0]["image_score"] == pytest.approx(1.0, abs=1e-3)

    def test_fusion_does_not_penalise_a_dual_match(self, store):
        """A two-modal match must outrank a stronger single-modality match.

        Under the old per-faiss_id fusion, item "a" scored 0.6*s_text and item
        "b" scored 0.6*s_text too, so a photo match could never help. Here "b"
        is a *perfect* text match with no photo, "a" is a good text match plus a
        perfect photo; fused, "a" must win because both its sides are real.
        """
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "text": "a", "vector": _vec(1)},
            {"faiss_id": 3, "item_id": "a", "modality": "image",
             "text": "", "vector": _vec(1)},
            {"faiss_id": 2, "item_id": "b", "modality": "text",
             "text": "b", "vector": _vec(1)},
        ])
        hits = store.search(text_vector=np.asarray(_vec(1)),
                            image_vector=np.asarray(_vec(1)), top_k=3,
                            fused=True, text_weight=0.5, image_weight=0.5)
        order = [h["item_id"] for h in hits]
        assert order.index("a") < order.index("b"), order

    def test_unfused_search_keeps_raw_cosine(self, store):
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "text": "x", "vector": _vec(1)},
            {"faiss_id": 3, "item_id": "a", "modality": "image",
             "text": "", "vector": _vec(1)},
        ])
        hits = store.search(text_vector=np.asarray(_vec(1)),
                            image_vector=np.asarray(_vec(1)), top_k=5, fused=False)
        assert {h["modality"] for h in hits} == {"text", "image"}
        assert all("matched_modalities" not in h for h in hits)

    def test_upsert_rejects_wrong_dimension_without_partial_write(self, store):
        store.upsert([{"faiss_id": 1, "item_id": "a", "modality": "text",
                       "text": "ok", "vector": _vec(1)}])
        with pytest.raises(ValueError, match="dim"):
            store.upsert([
                {"faiss_id": 2, "item_id": "b", "modality": "text",
                 "text": "good", "vector": _vec(2)},
                {"faiss_id": 3, "item_id": "c", "modality": "text",
                 "text": "bad", "vector": _vec(3, dim=16)},   # wrong width
            ])
        # The valid row in the same batch must not have landed.
        assert store.counts() == {"text": 1, "image": 0, "meta": 1}

    def test_upsert_rejects_non_finite_vectors(self, store):
        bad = _vec(1)
        bad[0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            store.upsert([{"faiss_id": 1, "item_id": "a", "modality": "text",
                           "text": "x", "vector": bad}])
        assert store.counts()["text"] == 0

    def test_upsert_rejects_unknown_modality(self, store):
        with pytest.raises(ValueError, match="modality"):
            store.upsert([{"faiss_id": 1, "item_id": "a", "modality": "audio",
                           "text": "x", "vector": _vec(1)}])

    def test_save_load_roundtrip(self, store):
        store.upsert([{"faiss_id": 9, "item_id": "z", "modality": "text",
                       "text": "hello", "vector": _vec(3)}])
        store.save()
        fresh = FaissStore(32, store.index_dir)
        fresh.load()
        hits = fresh.search(text_vector=np.asarray(_vec(3)), top_k=1)
        assert hits[0]["faiss_id"] == 9 and hits[0]["text"] == "hello"

    def test_dirty_tracking_and_autosave_interval(self, store):
        assert not store.dirty
        store.upsert([{"faiss_id": 1, "item_id": "a", "modality": "text",
                       "text": "x", "vector": _vec(1)}])
        assert store.dirty
        # First write always happens (nothing on disk yet).
        assert store.save_if_dirty(min_interval=3600) is True
        assert not store.dirty
        # A second write inside the interval is deferred, not written.
        store.upsert([{"faiss_id": 2, "item_id": "b", "modality": "text",
                       "text": "y", "vector": _vec(2)}])
        assert store.dirty
        assert store.save_if_dirty(min_interval=3600) is False

        # force overrides the interval
        assert store.save_if_dirty(min_interval=3600, force=True) is True
        assert not store.dirty


    def test_first_save_does_not_depend_on_host_uptime(self, store, monkeypatch):
        """Regression: the first autosave must not be gated on time.monotonic().

        `_last_saved` used to start at 0.0 while the interval check compared
        against time.monotonic(), which Linux measures from boot. On a host
        that had been up for less than min_interval the first save was silently
        skipped. This failed on a fresh CI runner and passed on a laptop.
        """
        import app.faiss_store as mod

        clock = {"t": 5.0}                      # pretend we booted 5s ago
        monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])

        store.upsert([{"faiss_id": 3, "item_id": "c", "modality": "text",
                       "text": "z", "vector": _vec(4)}])
        assert store.save_if_dirty(min_interval=3600) is True
        assert not store.dirty

        # and the interval is still honoured once a save has happened
        clock["t"] += 10
        store.upsert([{"faiss_id": 4, "item_id": "d", "modality": "text",
                       "text": "w", "vector": _vec(5)}])
        assert store.save_if_dirty(min_interval=3600) is False

        clock["t"] += 3601
        assert store.save_if_dirty(min_interval=3600) is True
    def test_index_survives_a_fresh_store_instance(self, store, tmp_path):
        """The regression that made every model restart empty the index."""
        store.upsert([{"faiss_id": 5, "item_id": "a", "modality": "text",
                       "text": "durable", "vector": _vec(4)}])
        store.save()
        revived = FaissStore(32, str(tmp_path))
        assert revived.counts() == {"text": 0, "image": 0, "meta": 0}
        revived.load()
        assert revived.counts()["text"] == 1
        assert revived.search(text_vector=np.asarray(_vec(4)), top_k=1)[0]["text"] == "durable"

    def test_item_filter(self, store):
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "text": "x", "vector": _vec(1)},
            {"faiss_id": 2, "item_id": "b", "modality": "text",
             "text": "y", "vector": _vec(1)},
        ])
        hits = store.search(text_vector=np.asarray(_vec(1)), top_k=5,
                            item_ids=["b"])
        assert [h["item_id"] for h in hits] == ["b"]


class TestAnswerComposer:
    def _ctxs(self):
        return [
            {"item_id": "a", "sku": "KX-1042", "modality": "text", "score": 0.9,
             "text": "The Kestrel bottle holds 12 oz and is dishwasher safe."},
            {"item_id": "a", "sku": "KX-1042", "modality": "image", "score": 0.7,
             "text": ""},
        ]

    def test_extractive_is_grounded_with_citations(self):
        comp = AnswerComposer(0.35)
        out = comp.compose("Is the Kestrel bottle dishwasher safe?", self._ctxs())
        assert out["citation_check"]["passed"] is True
        assert out["answer"] != NOT_GROUNDED
        assert out["sentences"] and all(s["citations"] for s in out["sentences"])
        assert out["sentences"][0]["citations"][0]["context_index"] == 0

    def test_no_evidence_refuses_to_answer(self):
        comp = AnswerComposer(0.35)
        out = comp.compose("What is the warranty on the lunar rover?",
                           [{"item_id": "a", "modality": "text", "score": 0.2,
                             "text": "The bottle is made of stainless steel."}])
        assert out["citation_check"]["passed"] is False
        assert out["answer"] == NOT_GROUNDED

    def test_abstractive_strips_unsupported_sentences(self):
        comp = AnswerComposer(0.35)
        out = comp.compose(
            "Is it dishwasher safe?", self._ctxs(), mode="abstractive",
            generator=lambda q, c: "The Kestrel bottle is dishwasher safe. "
                                   "It is also made of solid gold.")
        assert out["citation_check"]["passed"] is False  # gold claim stripped
        assert "gold" not in out["answer"]
        assert "dishwasher" in out["answer"]
        stripped = [d for d in out["citation_check"]["details"] if d.get("stripped")]
        assert len(stripped) == 1

    def test_empty_contexts(self):
        out = AnswerComposer(0.35).compose("anything?", [])
        assert out["answer"] == NOT_GROUNDED and not out["citation_check"]["passed"]

    def test_visual_question_cites_images(self):
        comp = AnswerComposer(0.35)
        out = comp.compose("What does the bottle look like?", self._ctxs())
        image_cited = any(c["context_index"] == 1
                          for s in out["sentences"] for c in s["citations"])
        assert image_cited
