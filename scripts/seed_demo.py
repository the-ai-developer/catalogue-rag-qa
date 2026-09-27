#!/usr/bin/env python3
"""Seed the Catalogue AI API with realistic demo data and traffic.

Creates 20 items across 5 categories (with optional product photos), triggers
ingest for each, then exercises Project 1 (3 grounded questions) and Project 2
(one spec → description job). Pure stdlib — runs anywhere.

**Idempotent.** SKUs are derived from the item title, so a second run reuses the
existing items instead of failing 20 times with 409 Conflict. --reset would
delete them first, but the api has no delete route, so it exits with an error
telling you to run `docker compose down -v` instead.

    python scripts/seed_demo.py --api-base http://localhost:8080 \
        --api-key local-dev-admin-key [--ingest-only] [--no-images] [--reset]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib

ITEMS = [
    ("drinkware", "Kestrel 12 oz Insulated Bottle", "18/8 stainless steel",
     {"height_cm": 26.0, "diameter_cm": 7.2, "capacity_oz": 12},
     ["double-wall vacuum insulation", "leak-proof lid", "dishwasher safe"]),
    ("drinkware", "Harbor 16 oz Travel Mug", "double-wall ceramic",
     {"height_cm": 18.0, "diameter_cm": 8.0, "capacity_oz": 16},
     ["one-hand open lid", "non-slip base", "dishwasher safe"]),
    ("drinkware", "Drift 24 oz Water Flask", "tritan copolyester",
     {"height_cm": 27.0, "capacity_oz": 24},
     ["BPA-free", "carry loop", "wide mouth"]),
    ("drinkware", "Atlas Espresso Cup Set", "stoneware",
     {"height_cm": 7.0, "capacity_oz": 4}, ["stackable", "oven safe", "non-toxic glaze"]),
    ("bags", "Ridge 40 L Travel Pack", "recycled nylon",
     {"height_cm": 55.0, "width_cm": 32.0}, ["laptop sleeve", "rain cover", "luggage pass-through"]),
    ("bags", "Summit 22 L Daypack", "waxed canvas",
     {"height_cm": 45.0, "width_cm": 28.0}, ["padded straps", "hidden pocket", "YKK zips"]),
    ("bags", "Drift Weekender Duffel", "full-grain leather",
     {"length_cm": 50.0, "width_cm": 28.0}, ["shoe compartment", "YKK zips", "shoulder strap"]),
    ("bags", "Kestrel Sling Bag", "recycled nylon",
     {"height_cm": 20.0, "width_cm": 15.0}, ["anti-theft zip", "quick-release buckle"]),
    ("apparel", "Harbor Training Tee", "organic cotton",
     {"chest_cm": 98.0, "length_cm": 72.0}, ["moisture-wicking", "flatlock seams", "anti-odour finish"]),
    ("apparel", "Atlas Merino Base Layer", "merino wool",
     {"chest_cm": 102.0, "length_cm": 74.0}, ["UPF 50+", "four-way stretch", "anti-odour finish"]),
    ("apparel", "Ridge Trail Shirt", "recycled polyester",
     {"chest_cm": 104.0, "length_cm": 76.0}, ["UPF 50+", "quick-dry", "ventilated back"]),
    ("apparel", "Summit Zip Hoodie", "organic cotton",
     {"chest_cm": 108.0, "length_cm": 78.0}, ["brushed interior", "kangaroo pocket"]),
    ("electronics", "Drift ANC Headphones", "anodised aluminium",
     {"width_cm": 18.0, "depth_cm": 8.0}, ["ANC", "40-hour battery", "multipoint pairing"]),
    ("electronics", "Atlas 20000 mAh Power Bank", "ABS+PC blend",
     {"width_cm": 8.0, "depth_cm": 2.5}, ["USB-C fast charge", "airline safe", "pass-through charging"]),
    ("electronics", "Harbor Smart Speaker", "recycled polycarbonate",
     {"height_cm": 12.0, "diameter_cm": 9.0}, ["Bluetooth 5.3", "IPX5 rated", "voice assistant"]),
    ("electronics", "Kestrel Earbuds Case", "anodised aluminium",
     {"width_cm": 6.0, "depth_cm": 2.5}, ["USB-C fast charge", "IPX5 rated", "pocket clip"]),
    ("home", "Ridge Serving Bowl", "solid oak",
     {"height_cm": 12.0, "width_cm": 30.0}, ["hand-finished edges", "easy-clean coating", "food safe"]),
    ("home", "Summit Storage Jar", "borosilicate glass",
     {"height_cm": 22.0, "diameter_cm": 12.0}, ["airtight lid", "stackable", "dishwasher safe"]),
    ("home", "Drift Chopping Board", "solid oak",
     {"length_cm": 45.0, "width_cm": 30.0}, ["juice groove", "hand-finished edges"]),
    ("home", "Atlas Table Lamp", "borosilicate glass",
     {"height_cm": 40.0, "width_cm": 18.0}, ["dimmable", "USB-C powered", "warm light"]),
]

QUESTIONS = [
    "Which bottle is dishwasher safe and what is its capacity?",
    "What material is the Ridge 40 L Travel Pack made from?",
    "Do any headphones offer ANC and how long is the battery life?",
]

TERMINAL = {"succeeded", "failed"}


def call(base: str, key: str, method: str, path: str, body=None, raw=None, headers=None):
    req_headers = {"X-API-Key": key}
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        req_headers["Content-Type"] = "application/json"
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(base + path, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, _decode(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())


def _decode(payload: bytes) -> dict:
    """Never raise on a non-JSON body.

    Pointing --api-base at a host that answers /readyz with an SPA index (nginx
    without a /readyz route) used to abort the whole seed with a traceback.
    """
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"_non_json": payload[:200].decode(errors="replace")}
    return parsed if isinstance(parsed, dict) else {"_list": parsed}


def sku_for(title: str) -> str:
    return f"SEED-{hashlib.sha256(title.encode()).hexdigest()[:8].upper()}"


def tiny_png() -> bytes:
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + ctype + data
                + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + b"\xb0\x80\x40" * 32 for _ in range(32))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def multipart(filename: str, content: bytes):
    boundary = "----seed" + uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += content + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def find_by_sku(base: str, key: str, sku: str):
    """Look an item up by exact SKU. listItems has no sku filter, so page until
    it turns up rather than creating a duplicate."""
    cursor = ""
    for _ in range(50):
        path = f"/api/v1/items?limit=100" + (f"&cursor={cursor}" if cursor else "")
        code, page = call(base, key, "GET", path)
        if code != 200:
            return None
        for item in page.get("items", []):
            if item.get("sku") == sku:
                return item
        cursor = page.get("next_cursor")
        if cursor in (None, "", 0):
            return None
    return None


def wait_for_ingest(base: str, key: str, job_ids: list, budget: int = 180) -> bool:
    """Block until every ingest job settles.

    Asking questions after a fixed sleep raced the workers and produced
    "no grounded evidence" on a cold stack.
    """
    pending = set(job_ids)
    deadline = time.time() + budget
    while pending and time.time() < deadline:
        for job_id in list(pending):
            code, job = call(base, key, "GET", f"/api/v1/jobs/ingest/{job_id}")
            if code != 200:
                pending.discard(job_id)
                continue
            if job.get("status") in TERMINAL:
                pending.discard(job_id)
                if job.get("status") == "failed":
                    print(f"  ! ingest {job_id[:8]} failed: {job.get('error')}",
                          file=sys.stderr)
        if pending:
            time.sleep(2)
    if pending:
        print(f"  ! {len(pending)} ingest job(s) still running after {budget}s",
              file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://localhost:8080")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY", "local-dev-admin-key"))
    ap.add_argument("--ingest-only", action="store_true")
    ap.add_argument("--no-images", dest="images", action="store_false",
                    help="skip product photos (much faster; skips the image index)")
    ap.add_argument("--reset", action="store_true",
                    help="delete existing demo items before recreating them "
                         "(needs DELETE /api/v1/items/{id}; the api does not "
                         "implement it yet, so use `docker compose down -v` instead)")
    ap.add_argument("--wait-secs", type=int, default=180)
    ap.set_defaults(images=True)
    args = ap.parse_args()
    base, key = args.api_base.rstrip("/"), args.api_key

    code, _ = call(base, key, "GET", "/readyz")
    if code != 200:
        print(f"API not ready at {base} (status {code})", file=sys.stderr)
        return 1

    if args.ingest_only:
        code, items = call(base, key, "GET", "/api/v1/items?limit=100")
        if code != 200:
            print(f"list items failed: {code} {items}", file=sys.stderr)
            return 1
        ids = [it["id"] for it in items.get("items", [])]
    else:
        ids = []
        created = reused = 0
        for cat, title, material, dims, features in ITEMS:
            sku = sku_for(title)
            if args.reset:
                existing = find_by_sku(base, key, sku)
                if existing:
                    code, body = call(base, key, "DELETE",
                                      f"/api/v1/items/{existing['id']}")
                    if code == 404:
                        # There is no delete route. Say so instead of continuing
                        # and leaving the caller to wonder why nothing changed.
                        print("--reset needs DELETE /api/v1/items/{id}, which this "
                              "api does not implement (404).\n"
                              "Recreate the stack with `docker compose down -v` to "
                              "start from an empty catalogue.", file=sys.stderr)
                        return 1
            item = find_by_sku(base, key, sku)
            if item is None:
                code, item = call(base, key, "POST", "/api/v1/items", {
                    "sku": sku, "title": title, "category": cat,
                    "material": material, "dimensions": dims, "features": features,
                    "extra": {"colour": "slate"}, "status": "active"})
                if code != 201:
                    print(f"  ! create failed ({code}): {item}", file=sys.stderr)
                    continue
                created += 1
            else:
                reused += 1
            ids.append(item["id"])
            # Only attach a photo when the item has none, so re-runs do not
            # accumulate duplicate assets.
            if args.images and not item.get("assets"):
                raw, headers = multipart(f"{item['id']}.png", tiny_png())
                call(base, key, "POST", f"/api/v1/items/{item['id']}/assets",
                     raw=raw, headers=headers)
        print(f"items: {created} created, {reused} reused, {len(ids)} total")

    jobs = []
    for item_id in ids:
        code, job = call(base, key, "POST", f"/api/v1/items/{item_id}/ingest", {})
        if code == 202 and job.get("job_id"):
            jobs.append(job["job_id"])
    print(f"enqueued {len(jobs)} ingest job(s); waiting…")
    if jobs and not wait_for_ingest(base, key, jobs, args.wait_secs):
        if not args.ingest_only:
            print("continuing anyway; answers may be ungrounded", file=sys.stderr)

    if args.ingest_only:
        return 0

    print("\n--- Project 1: grounded QA ---")
    for q in QUESTIONS:
        code, res = call(base, key, "POST", "/api/v1/qa/ask", {
            "question": q, "top_k": 6, "use_images": True,
            "composition": "extractive", "user_ref": "seed:buyer"})
        passed = res.get("citation_check", {}).get("passed")
        degraded = (res.get("retrieval") or {}).get("degraded")
        print(f"\nQ: {q}")
        print(f"A: {res.get('answer', res)}")
        print(f"   grounded={passed} citations={len(res.get('citations', []))} "
              f"latency={res.get('latency_ms', '?')}ms")
        if degraded:
            print(f"   ! retrieval degraded: {degraded.get('reason')} — "
                  "run `make train-projection`", file=sys.stderr)

    print("\n--- Project 2: spec → description (editor gate) ---")
    code, job = call(base, key, "POST", "/api/v1/descriptions/jobs", {
        "spec": {"title": "Kestrel 12 oz Insulated Bottle", "category": "drinkware",
                 "material": "18/8 stainless steel",
                 "dimensions": {"height_cm": 26.0, "capacity_oz": 12},
                 "features": ["double-wall vacuum insulation", "leak-proof lid"]},
        "beam_width": 4, "max_len": 192})
    if code != 202:
        print(f"  ! job create failed ({code}): {job}", file=sys.stderr)
        return 1
    job_id = job["job_id"]
    detail: dict = {}
    for _ in range(30):
        _, detail = call(base, key, "GET", f"/api/v1/descriptions/jobs/{job_id}")
        if detail.get("status") not in ("queued", "running"):
            break
        time.sleep(2)
    print(f"job {job_id[:8]} status={detail.get('status')}")
    for draft in detail.get("drafts", []):
        print(f"  [{draft['rank']}] ({draft['score']:.2f}) {draft['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
