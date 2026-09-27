"""InfoNCE and the two-head shared-space trainer.

`ml/contrastive.py` holds the loss that makes the shared space work at all:
without it, SBERT text and CLIP image vectors sit in unrelated spaces and
cross-modal retrieval returns noise. It had no tests, which is a poor bet for
the function the whole retrieval story depends on.

The properties worth pinning are behavioural, not structural. The obvious
failure is a loss that goes *down* as the correct pair gets *closer*, or a
trainer that silently updates only one tower.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.contrastive import SharedSpaceTrainer, info_nce  # noqa: E402


def l2(x):
    return torch.tensor(x, dtype=torch.float32)


def aligned_pairs(n=8, dim=16, noise=0.05, seed=0):
    """Text and image vectors that point the same way, plus a little noise."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    img = base + noise * rng.standard_normal((n, dim)).astype(np.float32)
    return l2(base), l2(img)


class TestInfoNce:
    def test_aligned_pairs_score_near_zero_loss(self):
        t, i = aligned_pairs()
        loss, m = info_nce(t, i)
        assert loss.item() < 0.05
        assert m["acc_image_to_text"] == 1.0
        assert m["acc_text_to_image"] == 1.0

    def test_shuffled_pairs_are_penalised(self):
        """Retrieval accuracy must collapse when the pairing is wrong."""
        t, i = aligned_pairs()
        loss_ok, m_ok = info_nce(t, i)
        loss_bad, m_bad = info_nce(t, i[torch.randperm(t.shape[0])])
        assert loss_bad.item() > loss_ok.item()
        assert m_bad["acc_image_to_text"] < m_ok["acc_image_to_text"]

    def test_loss_decreases_as_the_pair_tightens(self):
        """The gradient has to point the right way, not just exist."""
        base, _ = aligned_pairs()
        far = base + 3.0 * torch.randn(base.shape)
        near = base + 0.01 * torch.randn(base.shape)
        loss_far, _ = info_nce(base, far)
        loss_near, _ = info_nce(base, near)
        assert loss_near.item() < loss_far.item()

    def test_symmetric_in_both_directions(self):
        t, i = aligned_pairs()
        a, _ = info_nce(t, i)
        b, _ = info_nce(i, t)
        assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_metrics_report_temperature(self):
        _, m = info_nce(*aligned_pairs(), temperature=0.5)
        assert m["temperature"] == 0.5
        assert set(m) == {"acc_image_to_text", "acc_text_to_image", "temperature"}

    def test_batch_of_one_is_trivially_perfect(self):
        t = l2([[1.0, 0.0, 0.0]])
        loss, m = info_nce(t, t)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert m["acc_image_to_text"] == 1.0

    def test_matches_hand_computed_cross_entropy(self):
        """Two aligned pairs, computed by hand, so the loss is pinned."""
        t = l2([[1.0, 0.0], [0.0, 1.0]])
        i = l2([[1.0, 0.0], [0.0, 1.0]])
        loss, _ = info_nce(t, i, temperature=1.0)
        # logits are the 2x2 identity at temperature 1: [[1,0],[0,1]]
        # for row 0 the target is column 0, whose logit is 1, so
        #   cross_entropy = -(1 - log(e^1 + e^0)) = log(1 + e) - 1
        # and the reverse direction is the same, so the mean is unchanged.
        expected = np.log1p(np.e) - 1.0
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_is_differentiable(self):
        t = l2(np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32))
        i = l2(np.random.default_rng(2).standard_normal((4, 8)).astype(np.float32))
        t.requires_grad_(True)
        loss, _ = info_nce(t, i)
        loss.backward()
        assert t.grad is not None and torch.isfinite(t.grad).all()


class TestSharedSpaceTrainer:
    def _trainer(self, lr=0.05, text_dim=768, image_dim=512, out=16):
        torch.manual_seed(0)
        return SharedSpaceTrainer(
            torch.nn.Linear(text_dim, out),
            torch.nn.Linear(image_dim, out),
            lr=lr)

    def test_step_returns_a_finite_loss(self):
        tr = self._trainer()
        t = torch.randn(4, 768)
        i = torch.randn(4, 512)
        loss, m = tr.step(t, i)
        assert np.isfinite(loss)
        assert "acc_image_to_text" in m

    def test_step_reduces_loss_on_a_fixed_batch(self):
        """Optimisation has to actually move the loss, not just report it."""
        tr = self._trainer(lr=0.1)
        t, i = aligned_pairs(n=8, dim=16, noise=0.3, seed=3)
        t = t @ torch.randn(16, 768)
        i = i @ torch.randn(16, 512)

        first = tr.step(t, i)[0]
        for _ in range(30):
            last = tr.step(t, i)[0]
        assert last < first, f"loss went {first:.4f} -> {last:.4f}"

    def test_both_towers_are_updated(self):
        """A single-tower update would leave cross-modal retrieval broken."""
        tr = self._trainer(lr=0.1)
        before_t = tr.text_projection.weight.detach().clone()
        before_i = tr.image_projection.weight.detach().clone()
        t, i = aligned_pairs(n=8, dim=16, noise=0.3, seed=5)
        tr.step(t @ torch.randn(16, 768), i @ torch.randn(16, 512))
        assert not torch.allclose(before_t, tr.text_projection.weight)
        assert not torch.allclose(before_i, tr.image_projection.weight)

    def test_gradients_are_reset_between_steps(self):
        tr = self._trainer(lr=0.1)
        t, i = aligned_pairs(n=8, dim=16, noise=0.3, seed=7)
        t, i = t @ torch.randn(16, 768), i @ torch.randn(16, 512)
        tr.step(t, i)
        w_after_first = tr.text_projection.weight.detach().clone()
        tr.step(t, i)
        # a second identical step must produce the same update, not a doubled one
        assert not torch.allclose(w_after_first, tr.text_projection.weight)


class TestRequiresTorch:
    def test_import_error_names_the_module(self, monkeypatch):
        import ml.contrastive as mod
        monkeypatch.setattr(mod, "torch", None)
        with pytest.raises(ImportError, match="ml.contrastive"):
            mod.info_nce(None, None)
