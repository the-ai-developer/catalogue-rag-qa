"""Build the from-scratch ML notebooks (nbformat 4) — one source of truth.

    python ml/notebooks/build_notebooks.py

Keeps the .ipynb files valid JSON and consistent with the production
modules in services/model-server/app.
"""

from __future__ import annotations

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def nb(cells):
    return {
        "cells": [
            ({"cell_type": "markdown", "metadata": {},
              "source": md.splitlines(keepends=True)} if kind == "md" else
             {"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": code.splitlines(keepends=True)})
            for kind, body in cells for md, code in [(body, body)]
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NB1 = nb([
    ("md", "# 01 · Dual encoder from scratch\n\n"
           "Build a CLIP-style dual encoder (text tower + image tower + shared "
           "projection) with plain PyTorch on the synthetic product corpus. This is "
           "the maths behind `app/embed.py` + `app/projection.py` in production."),
    ("code", "import os, sys, json\n"
             "sys.path.insert(0, os.path.abspath('../..'))  # repo root\n"
             "sys.path.insert(0, os.path.abspath('../../services/model-server'))\n"
             "import torch, numpy as np\n"
             "from ml.contrastive import info_nce, SharedSpaceTrainer\n\n"
             "# CONFIG — scale up for real training\n"
             "CONFIG = dict(epochs=3, batch=32, dim=64, lr=1e-3, temperature=0.07, seed=42)\n"
             "torch.manual_seed(CONFIG['seed']); np.random.seed(CONFIG['seed'])\n"
             "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
             "print('device:', device)"),
    ("md", "## Data: hashed text + placeholder images as raw features\n\n"
           "We skip pretrained encoders here on purpose — the point is the "
           "**contrastive shared-space objective**, not the towers."),
    ("code", "from PIL import Image\n"
             "import csv\n\n"
             "manifest = '../../ml/data/generated/manifest.csv'\n"
             "rows = list(csv.DictReader(open(manifest)))[:512] if os.path.exists(manifest) else []\n"
             "if not rows:\n"
             "    raise SystemExit('run: python ml/data/make_synthetic_dataset.py first')\n\n"
             "def text_features(t):\n"
             "    v = np.zeros(CONFIG['dim'], np.float32)\n"
             "    for tok in t.lower().split():\n"
             "        v[hash(tok) % CONFIG['dim']] += 1.0\n"
             "    return v / (np.linalg.norm(v) + 1e-8)\n\n"
             "def image_features(path):\n"
             "    a = np.asarray(Image.open(path).resize((16, 16))).astype(np.float32) / 255\n"
             "    return a.reshape(-1)[:CONFIG['dim']] if a.size >= CONFIG['dim'] else np.pad(a.reshape(-1), (0, CONFIG['dim'] - a.size))\n\n"
             "T = torch.tensor(np.stack([text_features(r['description']) for r in rows]), device=device)\n"
             "I = torch.tensor(np.stack([image_features(os.path.join('../../ml/data/generated', r['image_path'])) for r in rows]), device=device)\n"
             "print('pairs:', tuple(T.shape), tuple(I.shape))"),
    ("md", "## Train the shared space with symmetric InfoNCE"),
    ("code", "proj = torch.nn.Linear(CONFIG['dim'], CONFIG['dim'], bias=False).to(device)\n"
             "trainer = SharedSpaceTrainer(proj, lr=CONFIG['lr'])\n"
             "history = []\n"
             "for epoch in range(CONFIG['epochs']):\n"
             "    perm = torch.randperm(len(T), device=device)\n"
             "    for i in range(0, len(T) - CONFIG['batch'], CONFIG['batch']):\n"
             "        idx = perm[i:i + CONFIG['batch']]\n"
             "        loss, metrics = trainer.step(T[idx], I[idx], CONFIG['temperature'])\n"
             "        history.append(loss)\n"
             "print('final loss:', history[-1])"),
    ("code", "import matplotlib.pyplot as plt\n"
             "plt.plot(history); plt.xlabel('step'); plt.ylabel('InfoNCE loss')\n"
             "plt.title('shared-space training'); plt.show()"),
    ("md", "## Retrieval demo: text → image and image → text"),
    ("code", "with torch.no_grad():\n"
             "    t = torch.nn.functional.normalize(proj(T), dim=-1)\n"
             "    i = torch.nn.functional.normalize(proj(I), dim=-1)\n"
             "scores = t @ i.t()\n"
             "print('image→text top1 acc:', (scores.argmax(1) == torch.arange(len(t), device=device)).float().mean().item())\n"
             "print('text→image top1 acc:', (scores.argmax(0) == torch.arange(len(t), device=device)).float().mean().item())\n"
             "print('sample ranks for item 0:', (scores[0] > scores[0, 0]).sum().item() + 1)"),
])

NB3 = nb([
    ("md", "# 03 · RAG evaluation: retrieval + grounded composition\n\n"
           "End-to-end check of Project 1 against a small fixture index built in "
           "notebook memory — same composer and FAISS store code as production."),
    ("code", "import os, sys, json\n"
             "sys.path.insert(0, os.path.abspath('../../services/model-server'))\n"
             "import numpy as np\n"
             "from app.faiss_store import FaissStore\n"
             "from app.answer_composer import AnswerComposer\n"
             "from app.embed import HashEmbedder\n\n"
             "emb = HashEmbedder(dim=64)\n"
             "store = FaissStore(dim=64)\n"
             "rows = [json.loads(l) for l in open('../../ml/data/generated/train.jsonl')][:120]"),
    ("code", "# index descriptions as chunk evidence\n"
             "entries = []\n"
             "for i, r in enumerate(rows):\n"
             "    vec = emb.encode([r['description']])[0]\n"
             "    entries.append(dict(faiss_id=i, item_id=r['item_id'], modality='text',\n"
             "                        text=r['description'], vector=vec.tolist()))\n"
             "store.upsert(entries)\n"
             "print('indexed:', store.counts())"),
    ("code", "# retrieval: ask about a known feature\n"
             "q = f\"Which item features {rows[0]['spec']['features'][0]}?\"\n"
             "hits = store.search(text_vector=emb.encode([q])[0], top_k=6)\n"
             "print('question:', q)\n"
             "print('top-1 item:', hits[0]['item_id'], 'gold:', rows[0]['item_id'], 'score:', round(hits[0]['score'], 3))"),
    ("code", "# grounded composition + citation gate\n"
             "composer = AnswerComposer(0.35)\n"
             "out = composer.compose(q, hits, mode='extractive')\n"
             "print('answer:', out['answer'])\n"
             "print('passed:', out['citation_check']['passed'])\n"
             "print('citations:', [(s['index'], s['citations']) for s in out['sentences']])"),
    ("code", "# the gate: an ungrounded question must refuse to answer\n"
             "bad = composer.compose('What is the warranty on the lunar rover?', hits)\n"
             "print('answer:', bad['answer'])\n"
             "assert bad['citation_check']['passed'] is False\n"
             "print('citation gate held ✔')"),
    ("code", "# hit@k / MRR over the generated qrels\n"
             "qrels = [json.loads(l) for l in open('../../ml/data/generated/qrels.jsonl')][:50]\n"
             "hit1 = rr = 0\n"
             "for q in qrels:\n"
             "    hs = store.search(text_vector=emb.encode([q['question']])[0], top_k=6)\n"
             "    rank = next((i + 1 for i, h in enumerate(hs) if h['item_id'] == q['item_id']), None)\n"
             "    hit1 += bool(rank and rank <= 1); rr += 1 / rank if rank else 0\n"
             "print(json.dumps({'hit@1': hit1 / len(qrels), 'mrr': rr / len(qrels)}, indent=2))"),
])


def main() -> None:
    for name, doc in [("01_dual_encoder_from_scratch.ipynb", NB1),
                      ("03_rag_evaluation.ipynb", NB3)]:
        path = os.path.join(OUT, name)
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1)
        json.load(open(path))  # validate
        print("wrote", path, f"({len(doc['cells'])} cells)")


if __name__ == "__main__":
    main()
