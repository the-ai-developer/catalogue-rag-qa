"""Tests: the HTTP surface — shared-secret auth, asset path confinement,
degradation reporting, and request validation.

Runs entirely on CPU with the deterministic hash embedder, so CI needs no model
weights and no GPU.
"""

import base64
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as ms
from app.embed import l2_normalise


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """A model-server with hash encoders, a writable asset root and no secret."""
    assets = tmp_path / "assets"
    (assets / "ab").mkdir(parents=True)
    (assets / "ab" / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    monkeypatch.setenv("EMBEDDER_BACKEND", "hash")
    monkeypatch.setenv("EMBED_DIM", "32")
    monkeypatch.setenv("ASSET_ROOT", str(assets))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "faiss"))
    monkeypatch.setenv("FAISS_AUTOSAVE_SECS", "0")  # no background thread in tests
    monkeypatch.setenv("MODEL_SHARED_SECRET", "")
    monkeypatch.setenv("DESC_MODEL_DIR", str(tmp_path / "missing-checkpoint"))

    ms.RT = ms.Runtime()
    with TestClient(ms.app) as c:
        yield c
    ms.RT = None


def _vec(seed: int, dim: int = 32) -> list:
    return l2_normalise(np.random.default_rng(seed).standard_normal(dim)).tolist()


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #

def test_open_when_no_secret_configured(client):
    assert client.get("/v1/models").status_code == 200


def test_401_without_secret(client, monkeypatch):
    monkeypatch.setenv("MODEL_SHARED_SECRET", "s3cr3t")
    ms.RT.settings = ms.get_settings()
    r = client.post("/v1/search", json={"query_text": "bottle"})
    assert r.status_code == 401


def test_401_with_wrong_secret(client, monkeypatch):
    monkeypatch.setenv("MODEL_SHARED_SECRET", "s3cr3t")
    ms.RT.settings = ms.get_settings()
    r = client.post("/v1/search", json={"query_text": "bottle"},
                    headers={"X-Model-Secret": "guess"})
    assert r.status_code == 401


def test_200_with_right_secret(client, monkeypatch):
    monkeypatch.setenv("MODEL_SHARED_SECRET", "s3cr3t")
    ms.RT.settings = ms.get_settings()
    r = client.post("/v1/search", json={"query_text": "bottle"},
                    headers={"X-Model-Secret": "s3cr3t"})
    assert r.status_code == 200


def test_health_endpoints_stay_open(client, monkeypatch):
    monkeypatch.setenv("MODEL_SHARED_SECRET", "s3cr3t")
    ms.RT.settings = ms.get_settings()
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)


# --------------------------------------------------------------------------- #
# asset path confinement
# --------------------------------------------------------------------------- #

def test_reads_a_relative_asset_path(client):
    r = client.post("/v1/embed/image", json={"images": [{"path": "ab/photo.png"}]})
    assert r.status_code == 200, r.text
    assert r.json()["dim"] == 32


@pytest.mark.parametrize("evil", [
    "../../../etc/passwd",
    "ab/../../../../etc/passwd",
    "./../../secret.env",
])
def test_refuses_paths_escaping_asset_root(client, evil):
    r = client.post("/v1/embed/image", json={"images": [{"path": evil}]})
    assert r.status_code in (400, 404), (evil, r.status_code, r.text)
    assert "escapes" in r.text or "not found" in r.text


def test_refuses_absolute_path_outside_root(client):
    r = client.post("/v1/embed/image",
                    json={"images": [{"path": "/etc/hostname"}]})
    assert r.status_code == 400
    assert "escapes ASSET_ROOT" in r.text


def test_missing_asset_is_404_not_500(client):
    r = client.post("/v1/embed/image", json={"images": [{"path": "ab/nope.png"}]})
    assert r.status_code == 404


def test_accepts_base64(client):
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"y" * 16).decode()
    r = client.post("/v1/embed/image", json={"images": [{"b64": png}]})
    assert r.status_code == 200, r.text


def test_rejects_malformed_base64(client):
    r = client.post("/v1/embed/image", json={"images": [{"b64": "!!!not base64!!!"}]})
    assert r.status_code == 400


def test_requires_path_or_b64(client):
    r = client.post("/v1/embed/image", json={"images": [{"asset_id": "x"}]})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def test_search_needs_a_query(client):
    assert client.post("/v1/search", json={}).status_code == 400


def test_top_k_is_bounded(client):
    assert client.post("/v1/search",
                       json={"query_text": "x", "top_k": 5000}).status_code == 422
    assert client.post("/v1/search",
                       json={"query_text": "x", "top_k": 0}).status_code == 422


def test_empty_text_batch_rejected(client):
    assert client.post("/v1/embed/text", json={"texts": []}).status_code == 422


def test_overlap_must_be_smaller_than_window(client):
    r = client.post("/v1/chunk", json={"text": "a b c", "max_tokens": 10,
                                       "overlap_tokens": 10})
    assert r.status_code == 400


def test_upsert_reports_bad_dimension_as_400(client):
    r = client.post("/v1/index/upsert", json={"entries": [
        {"faiss_id": 1, "item_id": "a", "modality": "text",
         "vector": [0.0] * 7}]})
    assert r.status_code == 400
    assert "dim" in r.text


# --------------------------------------------------------------------------- #
# degradation reporting
# --------------------------------------------------------------------------- #

def test_untrained_projection_is_reported_not_hidden(client):
    # No projection weights exist in the test fixture, so the shared space is
    # meaningless and the server must say so instead of serving noise.
    body = client.get("/v1/models").json()
    assert body["projection"]["trained"] is False
    assert body["degraded"]["reason"] == "untrained_projection"

    ready = client.get("/readyz").json()
    assert ready["ok"] is False
    assert ready["degraded"]["reason"] == "untrained_projection"

    hits = client.post("/v1/search", json={"query_text": "bottle"}).json()
    assert hits["degraded"]["reason"] == "untrained_projection"


def test_trained_projection_is_not_flagged(client, tmp_path, monkeypatch):
    from app.projection import ProjectionMapper
    rng = np.random.default_rng(0)
    ms.RT.projection = ProjectionMapper(
        32, text_weight=rng.standard_normal((32, 32)).astype(np.float32),
        image_weight=rng.standard_normal((32, 32)).astype(np.float32),
        trained=True)
    assert ms.RT.degraded() is None
    assert client.get("/readyz").json()["ok"] is True
    assert client.get("/v1/models").json()["degraded"] is None
