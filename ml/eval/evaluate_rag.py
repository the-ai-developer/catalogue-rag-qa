"""Retrieval eval for Project 1: hit@k, MRR and groundedness over qrels.

    python ml/eval/evaluate_rag.py --qrels ml/data/generated/qrels.jsonl \
        --index http://localhost:8090          # live model-server index
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def ask_index(base: str, question: str, top_k: int) -> list:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/search",
        data=json.dumps({"query_text": question, "top_k": top_k}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read()).get("hits", [])


def metrics(qrels: list, hits_by_q: list, ks=(1, 3, 6)) -> dict:
    hit = {k: 0 for k in ks}
    rr_total = 0.0
    for q, hits in zip(qrels, hits_by_q):
        gold = q["item_id"]
        ranks = [i + 1 for i, h in enumerate(hits) if h.get("item_id") == gold]
        for k in ks:
            if ranks and ranks[0] <= k:
                hit[k] += 1
        if ranks:
            rr_total += 1.0 / ranks[0]
    n = max(1, len(qrels))
    out = {f"hit@{k}": round(hit[k] / n, 4) for k in ks}
    out["mrr"] = round(rr_total / n, 4)
    out["n"] = len(qrels)
    return out


def groundedness_report(hits_by_q: list, threshold: float = 0.35) -> dict:
    """Share of questions whose top hit is strong enough to ground an answer."""
    strong = sum(1 for hits in hits_by_q if hits and hits[0].get("score", 0) >= threshold)
    total = sum(1 for hits in hits_by_q if hits)
    return {"strong_top_hit_rate": round(strong / max(1, total), 4),
            "empty_retrieval": sum(1 for hits in hits_by_q if not hits)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--index", default="http://localhost:8090")
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    with open(args.qrels) as fh:
        qrels = [json.loads(line) for line in fh]
    if not qrels:
        raise SystemExit(f"{args.qrels} is empty")
    hits = [ask_index(args.index, q["question"], args.top_k) for q in qrels]
    print(json.dumps({"retrieval": metrics(qrels, hits),
                      "groundedness": groundedness_report(hits)}, indent=2))


if __name__ == "__main__":
    main()
