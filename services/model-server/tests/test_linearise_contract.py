"""The spec linearisation is a wire format, so it is pinned exactly.

``linearise_spec`` is duplicated in this repository and in
`spec-to-description` (the T5 fine-tune imports it to build its inputs). Two
copies means two chances to drift, and a drift here is silent: the serving
model server would compose answers against one format while the model was
trained on another, and the only symptom would be a slightly worse score with
no error anywhere.

The assertions below are exact string comparisons, and the same file is kept in
both repositories. Change the format in one and this fails in both, which is
the intent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.linearise import linearise_spec  # noqa: E402

GOLDEN = {
    "title": "Kestrel 12 oz Insulated Bottle",
    "category": "drinkware",
    "material": "18/8 stainless steel",
    "dimensions": {"height_cm": 26.0, "capacity_oz": 12},
    "features": ["double-wall vacuum insulation", "leak-proof lid"],
}


class TestFormat:
    def test_full_spec_golden(self):
        # fixed field order: category | material | dimensions(sorted) |
        # features(order kept) | title | extra(sorted)
        assert linearise_spec(GOLDEN) == (
            "spec: category=drinkware | material=18/8 stainless steel | "
            "dimensions: capacity_oz=12, height_cm=26.0 | "
            "features: double-wall vacuum insulation, leak-proof lid | "
            "title: Kestrel 12 oz Insulated Bottle")

    def test_missing_fields_omitted(self):
        assert linearise_spec({"category": "bags"}) == "spec: category=bags"
        assert linearise_spec({"material": "nylon", "category": "bags"}) == \
            "spec: category=bags | material=nylon"

    def test_extra_sorted_and_deterministic(self):
        spec = {"category": "x", "extra": {"zeta": 1, "alpha": ["a", "b"]}}
        assert linearise_spec(spec) == "spec: category=x | extra: alpha=[a, b], zeta=1"

    def test_prefix_and_separator(self):
        out = linearise_spec({"category": "a", "material": "b"})
        assert out.startswith("spec: ")
        assert " | " in out

    def test_whitespace_is_collapsed(self):
        assert linearise_spec({"material": "  nylon \n  ripstop "}) == \
            "spec: material=nylon ripstop"


class TestValueRendering:
    @pytest.mark.parametrize("value,expected", [
        (True, "true"),
        (False, "false"),
        (12, "12"),
        (26.0, "26.0"),
        (["a", "b"], "[a, b]"),
    ])
    def test_scalars_and_lists(self, value, expected):
        assert linearise_spec({"extra": {"k": value}}) == \
            f"spec: extra: k={expected}"

    def test_nested_mapping(self):
        out = linearise_spec({"dimensions": {"b": 2, "a": 1}})
        assert out == "spec: dimensions: a=1, b=2"

    def test_empty_spec_is_just_the_prefix(self):
        assert linearise_spec({}) == "spec: "


class TestStability:
    def test_key_order_does_not_change_the_output(self):
        a = linearise_spec({"title": "T", "category": "c", "material": "m"})
        b = linearise_spec({"material": "m", "title": "T", "category": "c"})
        assert a == b

    def test_feature_order_is_preserved(self):
        """Features are a set of claims; their order carries meaning for the model."""
        out = linearise_spec({"features": ["zebra", "apple"]})
        assert "zebra, apple" in out
