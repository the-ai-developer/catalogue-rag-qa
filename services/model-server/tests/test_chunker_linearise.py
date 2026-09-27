"""Tests: chunking + normative spec linearisation (CPU, no downloads)."""

from app.chunker import Tokenizer, chunk_text
from app.linearise import linearise_spec


class TestChunker:
    def test_empty(self):
        assert chunk_text("   ") == []

    def test_deterministic_ordinals_and_overlap(self):
        text = ("The bottle is dishwasher safe. It holds twelve ounces. "
                "The lid is leak-proof and BPA-free. "
                "It keeps drinks cold for twenty-four hours. "
                "The shell is double-wall stainless steel.")
        a = chunk_text(text, max_tokens=12, overlap_tokens=4,
                       tokenizer=Tokenizer())
        b = chunk_text(text, max_tokens=12, overlap_tokens=4,
                       tokenizer=Tokenizer())
        assert [c.to_dict() for c in a] == [c.to_dict() for c in b]
        assert [c.ordinal for c in a] == list(range(len(a)))
        assert len(a) > 1
        # no text lost: every sentence's words appear in some chunk
        joined = " ".join(c.text for c in a)
        for word in text.split():
            assert word in joined

    def test_oversized_sentence_hard_split(self):
        text = " ".join(f"token{i}" for i in range(100))
        chunks = chunk_text(text, max_tokens=20, overlap_tokens=5)
        assert len(chunks) >= 5
        assert all(c.token_count <= 25 for c in chunks)


class TestLinearise:
    def test_full_spec_golden(self):
        spec = {"title": "Kestrel 12 oz Insulated Bottle",
                "category": "drinkware", "material": "18/8 stainless steel",
                "dimensions": {"height_cm": 26.0, "capacity_oz": 12},
                "features": ["double-wall vacuum insulation", "leak-proof lid"]}
        out = linearise_spec(spec)
        # fixed field order: category | material | dimensions(sorted) |
        # features(order kept) | title | extra(sorted)
        assert out == ("spec: category=drinkware | material=18/8 stainless steel | "
                       "dimensions: capacity_oz=12, height_cm=26.0 | "
                       "features: double-wall vacuum insulation, leak-proof lid | "
                       "title: Kestrel 12 oz Insulated Bottle")

    def test_missing_fields_omitted(self):
        assert linearise_spec({"category": "bags"}) == "spec: category=bags"
        assert linearise_spec({"material": "nylon", "category": "bags"}) == \
            "spec: category=bags | material=nylon"

    def test_extra_sorted_and_deterministic(self):
        spec = {"category": "x", "extra": {"zeta": 1, "alpha": ["a", "b"]}}
        out = linearise_spec(spec)
        assert out == "spec: category=x | extra: alpha=[a, b], zeta=1"
