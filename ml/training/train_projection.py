"""Train the shared-space projection (SBERT text + CLIP image → one space).

    python ml/training/train_projection.py --manifest ml/data/generated/manifest.csv \
        --out ml/checkpoints/projection/projection.pt

Symmetric InfoNCE over (description text, product image) pairs with in-batch
negatives, using the from-scratch implementation in
``ml/contrastive.py`` so the notebook maths and the production
trainer are the same code.

**Two heads, because the encoders disagree on width.** ``all-mpnet-base-v2``
emits 768 dims and ``clip-vit-base-patch32``'s image tower emits 512, so a
single shared matrix cannot consume both. The trainer therefore learns two
bias-free maps into the same ``--dim`` space (768→512 and 512→512 by default);
that is exactly what ``app/projection.py`` serves. The previous version
asserted the two raw dimensions were equal, which never held for the documented
default models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "services", "model-server"))

import numpy as np  # noqa: E402


def load_pairs(path: str):
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def batch_slices(n: int, batch: int):
    """Yield contiguous batches, never zero of them.

    ``range(0, n - batch + 1, batch)`` is empty whenever n < batch, which made
    small manifests (including a fresh ``--n 40`` synthetic set) train on
    nothing and then crash on an undefined metric.
    """
    if n == 0:
        return
    step = max(1, min(batch, n))
    for start in range(0, n, step):
        yield list(range(start, min(start + step, n)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="ml/checkpoints/projection/projection.pt")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from app.embed import ImageEmbedder, TextEmbedder, l2_normalise
    from ml.contrastive import SharedSpaceTrainer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    pairs = load_pairs(args.manifest)
    if not pairs:
        raise SystemExit(f"manifest {args.manifest} has no rows")
    root = os.path.dirname(args.manifest)
    text_enc = TextEmbedder(device=str(device))
    image_enc = ImageEmbedder(device=str(device))

    texts = [p["description"] for p in pairs]
    images = [os.path.join(root, p["image_path"]) for p in pairs]
    print(json.dumps({"pairs": len(pairs), "device": str(device)}), flush=True)

    text_vecs = l2_normalise(text_enc.encode(texts))
    image_vecs = l2_normalise(image_enc.encode(images))
    text_dim = int(text_vecs.shape[1])
    image_dim = int(image_vecs.shape[1])

    text_head = torch.nn.Linear(text_dim, args.dim, bias=False).to(device)
    image_head = torch.nn.Linear(image_dim, args.dim, bias=False).to(device)
    trainer = SharedSpaceTrainer(text_head, image_head, lr=args.lr)

    history = []
    metrics = {}
    n = len(pairs)
    for epoch in range(args.epochs):
        order = list(range(n))
        random.shuffle(order)
        losses = []
        for idx in batch_slices(n, args.batch):
            t = torch.tensor(text_vecs[idx], device=device)
            i = torch.tensor(image_vecs[idx], device=device)
            loss, metrics = trainer.step(t, i, args.temperature)
            losses.append(loss)
        event = {"epoch": epoch,
                 "loss": round(float(np.mean(losses)), 4) if losses else None,
                 "batches": len(losses), **metrics}
        history.append(event)
        print(json.dumps(event), flush=True)

    directory = os.path.dirname(args.out) or "."
    os.makedirs(directory, exist_ok=True)
    tw = text_head.weight.detach().cpu().numpy().astype("float32")
    iw = image_head.weight.detach().cpu().numpy().astype("float32")
    np.savez(args.out.rsplit(".", 1)[0] + ".npz", text=tw, image=iw)
    torch.save({"text": torch.from_numpy(tw), "image": torch.from_numpy(iw)}, args.out)
    with open(os.path.join(directory, "train_meta.json"), "w") as fh:
        json.dump({"text_dim_in": text_dim, "image_dim_in": image_dim,
                   "dim_out": args.dim, "temperature": args.temperature,
                   "pairs": n, "history": history}, fh)
    print(json.dumps({"saved": args.out, "text_dim_in": text_dim,
                      "image_dim_in": image_dim, "dim_out": args.dim}))


if __name__ == "__main__":
    main()
