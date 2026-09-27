"""Projection mapper and the hash embedder used when nothing is trained."""

import numpy as np
import pytest

from app.embed import HashEmbedder, l2_normalise
from app.projection import ProjectionMapper


class TestProjection:
    def test_identity_maps_and_normalises(self):
        proj = ProjectionMapper(embed_dim=8, trained=False)
        x = np.ones((2, 8), np.float32)
        y = proj.project_text(x)
        assert y.shape == (2, 8)
        assert np.allclose(np.linalg.norm(y, axis=1), 1.0)

    def test_two_towers_may_have_different_input_dims(self):
        """The real config: SBERT 768 in, CLIP 512 in, both out at 512."""
        rng = np.random.default_rng(0)
        proj = ProjectionMapper(
            embed_dim=16,
            text_weight=rng.standard_normal((16, 768)).astype(np.float32),
            image_weight=rng.standard_normal((16, 512)).astype(np.float32),
            trained=True)
        t = proj.project_text(rng.standard_normal((3, 768)).astype(np.float32))
        i = proj.project_image(rng.standard_normal((2, 512)).astype(np.float32))
        assert t.shape == (3, 16) and i.shape == (2, 16)
        assert np.allclose(np.linalg.norm(t, axis=1), 1.0)
        assert np.allclose(np.linalg.norm(i, axis=1), 1.0)

    def test_save_load_roundtrip_two_headed(self, tmp_path):
        rng = np.random.default_rng(1)
        proj = ProjectionMapper(
            embed_dim=4,
            text_weight=rng.standard_normal((4, 8)).astype(np.float32),
            image_weight=rng.standard_normal((4, 5)).astype(np.float32),
            trained=True)
        path = str(tmp_path / "projection.pt")
        proj.save(path)
        loaded = ProjectionMapper.load(path, embed_dim=4)
        assert loaded.trained and not loaded.legacy
        assert loaded.text_weight.shape == (4, 8)
        assert loaded.image_weight.shape == (4, 5)
        assert np.allclose(loaded.project_text(np.ones(8, np.float32)),
                           proj.project_text(np.ones(8, np.float32)), atol=1e-6)

    def test_legacy_single_matrix_is_flagged(self, tmp_path):
        """A one-matrix checkpoint still loads, but callers can be told."""
        w = np.eye(4, 8, dtype=np.float32)
        path = str(tmp_path / "projection.pt")
        np.savez(path.rsplit(".", 1)[0] + ".npz", weight=w)
        loaded = ProjectionMapper.load(path, embed_dim=4)
        assert loaded.trained and loaded.legacy
        assert loaded.info()["legacy_single_matrix"] is True

    def test_refuses_to_save_untrained(self, tmp_path):
        with pytest.raises(ValueError, match="untrained"):
            ProjectionMapper(embed_dim=4, trained=False).save(
                str(tmp_path / "p.pt"))

    def test_missing_weights_report_untrained(self, tmp_path):
        loaded = ProjectionMapper.load(str(tmp_path / "nope.pt"), embed_dim=6)
        assert not loaded.trained
        assert loaded.info()["trained"] is False
        assert loaded.project_text(np.ones(6, np.float32)).shape == (6,)

    def test_unreadable_weights_report_untrained_not_wrong(self, tmp_path):
        path = str(tmp_path / "projection.pt")
        (tmp_path / "projection.npz").write_bytes(b"not a real npz")
        loaded = ProjectionMapper.load(path, embed_dim=6)
        assert not loaded.trained


class TestHashEmbedders:
    def test_deterministic_and_normalised(self):
        emb = HashEmbedder(dim=16)
        a = emb.encode(["hello", "world"])
        b = emb.encode(["hello", "world"])
        assert a.shape == (2, 16)
        assert np.allclose(a, b)
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
        # unrelated inputs should not collapse to the same vector
        assert float(a[0] @ a[1]) < 0.9

    def test_bytes_input(self):
        emb = HashEmbedder(dim=8)
        out = emb.encode([b"\x89PNG"])
        assert out.shape == (1, 8)
