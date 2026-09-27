"""From-scratch InfoNCE contrastive learning for the shared vector space.

This is the training objective behind the SBERT+CLIP shared-space projection
(``ml/training/train_projection.py``): paired (text, image) items should be
close, all in-batch non-pairs far apart — symmetric in both directions.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = F = None


def _require_torch():
    if torch is None:
        raise ImportError("torch is required for ml.contrastive")


def info_nce(text_embeds, image_embeds, temperature: float = 0.07) -> Tuple["torch.Tensor", dict]:
    """Symmetric InfoNCE over a batch of aligned (text_i, image_i) pairs.

    Both inputs must be L2-normalised ``[B, D]``.  Returns (loss, metrics) where
    metrics includes retrieval accuracy in both directions.
    """
    _require_torch()
    logits = (text_embeds @ image_embeds.t()) / temperature  # [B,B]
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_i2t + loss_t2i)
    with torch.no_grad():
        acc_i2t = (logits.argmax(dim=1) == labels).float().mean()
        acc_t2i = (logits.argmax(dim=0) == labels).float().mean()
    return loss, {"acc_image_to_text": float(acc_i2t),
                  "acc_text_to_image": float(acc_t2i),
                  "temperature": temperature}


class SharedSpaceTrainer:
    """One optimisation step of the shared-space projection.

    Two heads: the towers have different input widths (SBERT 768, CLIP 512), so
    a single module cannot consume both. Both are mapped into the same
    ``dim_out`` space and the InfoNCE is computed there, which is what makes
    text↔image retrieval meaningful.
    """

    def __init__(self, text_projection: "torch.nn.Module",
                 image_projection: "torch.nn.Module", lr: float = 1e-3):
        _require_torch()
        self.text_projection = text_projection
        self.image_projection = image_projection
        self.optim = torch.optim.AdamW(
            list(text_projection.parameters()) + list(image_projection.parameters()),
            lr=lr)

    def step(self, text_raw, image_raw, temperature: float = 0.07):
        self.optim.zero_grad()
        t = F.normalize(self.text_projection(text_raw), dim=-1)
        i = F.normalize(self.image_projection(image_raw), dim=-1)
        loss, metrics = info_nce(t, i, temperature)
        loss.backward()
        self.optim.step()
        return float(loss.detach()), metrics
