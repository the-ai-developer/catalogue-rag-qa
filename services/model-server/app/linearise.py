"""NORMATIVE spec linearisation (API contract § "Spec linearisation").

Python owns this function; Go sends structured specs only.  The canonical
rendering is::

    spec: category=drinkware | material=18/8 stainless steel |
          dimensions: height_cm=26.0, capacity_oz=12 |
          features: double-wall vacuum insulation, leak-proof lid |
          title: Kestrel 12 oz Insulated Bottle

Field order is fixed: ``category``, ``material``, ``dimensions`` (keys sorted),
``features`` (given order), ``title``, ``extra`` (keys sorted).  Missing fields
are omitted entirely (including their separator).  ``extra`` key/value pairs use
the same ``key=value`` shape as ``dimensions``; list values render as
``[a, b]``.  Numeric values render with ``str()`` (``26.0`` stays ``26.0``,
``12`` stays ``12``).

Note: the contract's inline example prints ``height_cm`` before
``capacity_oz``; the normative rule is "dimensions sorted by key", which this
implementation follows (``capacity_oz`` sorts before ``height_cm``).
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

__all__ = ["linearise_spec"]

_PREFIX = "spec: "
_SEPARATOR = " | "


def _render_value(value: Any) -> str:
    """Render a scalar or list value deterministically as a string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Sequence):
        return "[" + ", ".join(_render_value(v) for v in value) + "]"
    if isinstance(value, Mapping):
        inner = ", ".join(
            f"{k}={_render_value(value[k])}" for k in sorted(value.keys())
        )
        return "{" + inner + "}"
    return str(value)


def _kv_block(name: str, mapping: Mapping[str, Any]) -> str:
    """Render ``name: k=v, k=v`` with keys sorted; empty mapping → empty string."""
    if not mapping:
        return ""
    pairs = ", ".join(
        f"{key}={_render_value(mapping[key])}" for key in sorted(mapping.keys())
    )
    return f"{name}: {pairs}"


def linearise_spec(spec: Mapping[str, Any]) -> str:
    """Linearise a structured spec sheet into the canonical single-line form.

    Args:
        spec: Mapping with any of ``category``, ``material``, ``dimensions``,
            ``features``, ``title``, ``extra``.  Unknown keys are ignored.

    Returns:
        The deterministic linearised string, always starting with ``spec: ``.
    """
    segments: list[str] = []

    category = spec.get("category")
    if category not in (None, ""):
        segments.append(f"category={_render_value(category)}")

    material = spec.get("material")
    if material not in (None, ""):
        segments.append(f"material={_render_value(material)}")

    dims = spec.get("dimensions")
    if isinstance(dims, MutableMapping) and dims:
        block = _kv_block("dimensions", dims)
        if block:
            segments.append(block)

    features = spec.get("features")
    if isinstance(features, Sequence) and not isinstance(features, str) and features:
        rendered = ", ".join(_render_value(f) for f in features)
        if rendered:
            segments.append(f"features: {rendered}")

    title = spec.get("title")
    if title not in (None, ""):
        segments.append(f"title: {_render_value(title)}")

    extra = spec.get("extra")
    if isinstance(extra, MutableMapping) and extra:
        block = _kv_block("extra", extra)
        if block:
            segments.append(block)

    return _PREFIX + _SEPARATOR.join(segments)
