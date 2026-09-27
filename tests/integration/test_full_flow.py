"""Cross-service integration tests for Catalogue AI.

Runs against a LIVE stack (docker compose up): Go API + model-server + Postgres.
Pure stdlib so it needs no install:  python3 -m pytest tests/integration -q
or directly:                            python3 tests/integration/test_full_flow.py

Env:
    API_BASE   (default http://localhost:8080)
    API_KEY    (admin key; default matches the local bootstrap key in .env.example)
    EDITOR_KEY (editor key; defaults to API_KEY)
    VIEWER_KEY (viewer key; defaults to API_KEY)
    WAIT_SECS  (per-job poll budget, default 120)
"""

from __future__ import annotations

import base64
import io
import json
import os
import struct
import sys
import time
import unittest
import urllib.error
import urllib.request
import zlib
import uuid

API_BASE = os.environ.get("API_BASE", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "local-dev-admin-key")
# Distinct by default: aliasing them to API_KEY silently turned the role tests
# into no-ops (a "viewer" that is really an admin can do anything).
EDITOR_KEY = os.environ.get("EDITOR_KEY", "local-dev-editor-key")
VIEWER_KEY = os.environ.get("VIEWER_KEY", "local-dev-viewer-key")
WAIT_SECS = int(os.environ.get("WAIT_SECS", "120"))

# Kept as a constant so the stored-answer assertion can compare against it.
QUESTION = ("Is the Kestrel bottle dishwasher safe "
            "and what is its capacity?")


# --------------------------------------------------------------------------- #
# tiny stdlib HTTP client + PNG factory
# --------------------------------------------------------------------------- #

def _call(method: str, path: str, *, body=None, key=API_KEY, raw=None,
          headers=None, timeout=60):
    url = API_BASE + path
    data = raw
    hdrs = {"X-API-Key": key}
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            code = resp.status
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        code = exc.code
    try:
        parsed = json.loads(payload) if payload else {}
    except json.JSONDecodeError:
        parsed = {"_raw": payload.decode(errors="replace")}
    return code, parsed


def tiny_png(width: int = 8, height: int = 8, rgb=(200, 120, 40)) -> bytes:
    """Minimal valid PNG built with stdlib only."""
    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + ctype + data
                + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    idat = chunk(b"IDAT", zlib.compress(raw))
    return sig + ihdr + idat + chunk(b"IEND", b"")


def multipart_file(field: str, filename: str, content: bytes,
                   mime: str = "image/png"):
    boundary = "----cat" + uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(f'Content-Disposition: form-data; name="{field}"; '
               f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
    body.write(content)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    return body.getvalue(), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def wait_job(path: str, terminal=("succeeded", "failed", "draft_ready",
                                 "approved", "rejected", "published")):
    deadline = time.time() + WAIT_SECS
    last = {}
    while time.time() < deadline:
        code, last = _call("GET", path)
        assert code == 200, f"{path} -> {code} {last}"
        if last.get("status") in terminal:
            return last
        time.sleep(2)
    raise AssertionError(f"job {path} not terminal within {WAIT_SECS}s: {last}")


def make_item() -> dict:
    code, item = _call("POST", "/api/v1/items", body={
        "sku": f"IT-{uuid.uuid4().hex[:10].upper()}",
        "title": "Kestrel 12 oz Insulated Bottle",
        "category": "drinkware",
        "material": "18/8 stainless steel",
        "dimensions": {"height_cm": 26.0, "diameter_cm": 7.2, "capacity_oz": 12},
        "features": ["double-wall vacuum insulation", "leak-proof lid", "BPA-free"],
        "extra": {"colours": ["slate", "sand"]},
    })
    assert code == 201, f"create item -> {code} {item}"
    return item


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

class TestPlatform(unittest.TestCase):
    def test_model_server_reports_degradation_or_health(self):
        """The model server must be reachable and honest about its state.

        An untrained projection makes cross-modal scores meaningless, which the
        server must report rather than serve as if it were a working shared
        space. Either state is acceptable here; silence is not.
        """
        code, body = _call("POST", "/api/v1/qa/ask", key=VIEWER_KEY,
                           body={"question": "anything?", "top_k": 1})
        if code == 200:
            degraded = (body.get("retrieval") or {}).get("degraded")
            if degraded:
                self.assertIn("reason", degraded)
                self.assertIn("detail", degraded)
        elif code not in (502, 503):
            self.fail(f"unexpected status {code}: {body}")

    def test_health_and_ready(self):
        code, body = _call("GET", "/healthz", key="")
        self.assertEqual(code, 200)
        code, body = _call("GET", "/readyz", key="")
        self.assertEqual(code, 200, f"readyz -> {body}")

    def test_auth_envelope(self):
        code, body = _call("GET", "/api/v1/items", key="wrong-key")
        self.assertEqual(code, 401)
        self.assertEqual(body.get("error", {}).get("code"), "unauthorized")

    def test_error_envelope_shape(self):
        code, body = _call("GET", "/api/v1/items/00000000-0000-0000-0000-000000000000")
        self.assertEqual(code, 404)
        self.assertEqual(body.get("error", {}).get("code"), "not_found")

    def test_viewer_cannot_create(self):
        if VIEWER_KEY == API_KEY:
            self.skipTest("VIEWER_KEY must differ from API_KEY to test the role")
        code, body = _call("POST", "/api/v1/items", key=VIEWER_KEY,
                           body={"sku": f"RB-{uuid.uuid4().hex[:8]}",
                                 "title": "t", "category": "c"})
        self.assertIn(code, (401, 403), body)

    def test_editor_cannot_create_items_but_can_ask(self):
        """The role boundary is catalogue-write (admin) vs editorial work."""
        if EDITOR_KEY == API_KEY:
            self.skipTest("EDITOR_KEY must differ from API_KEY")
        code, body = _call("POST", "/api/v1/items", key=EDITOR_KEY,
                           body={"sku": f"RB-{uuid.uuid4().hex[:8]}",
                                 "title": "t", "category": "c"})
        self.assertIn(code, (401, 403), body)
        code, _ = _call("POST", "/api/v1/qa/ask", key=VIEWER_KEY,
                        body={"question": "anything?", "top_k": 1})
        self.assertIn(code, (200, 502, 503))


class TestCatalogue(unittest.TestCase):
    def test_crud_and_validation(self):
        item = make_item()
        self.assertEqual(item["title"], "Kestrel 12 oz Insulated Bottle")
        self.assertEqual(item["status"], "draft")

        code, got = _call("GET", f"/api/v1/items/{item['id']}")
        self.assertEqual(code, 200)
        self.assertEqual(got["sku"], item["sku"])

        code, upd = _call("PATCH", f"/api/v1/items/{item['id']}",
                          body={"status": "active", "material": "titanium"})
        self.assertEqual(code, 200)
        self.assertEqual(upd["status"], "active")

        code, bad = _call("POST", "/api/v1/items", body={"title": "no sku"})
        self.assertEqual(code, 400)
        self.assertEqual(bad["error"]["code"], "bad_request")

        code, _ = _call("DELETE", f"/api/v1/items/{item['id']}")
        self.assertEqual(code, 204)
        code, archived = _call("GET", f"/api/v1/items/{item['id']}")
        self.assertIn(code, (200, 404))
        if code == 200:
            self.assertEqual(archived["status"], "archived")

    def test_optional_fields_never_round_trip_as_null(self):
        """Regression: an item created without features stored jsonb `null`.

        marshal() only guarded `v == nil`, which misses a *typed* nil inside an
        interface, so the column's NOT NULL DEFAULT '[]' was bypassed. The read
        path then returned `"features": null` and ItemPage's
        `features.join(', ')` threw — leaving an eternal "Loading item…".
        """
        code, item = _call("POST", "/api/v1/items", body={
            "sku": f"BARE-{uuid.uuid4().hex[:8]}", "title": "Bare Item",
            "category": "home"})
        self.assertEqual(code, 201, item)

        for path in (f"/api/v1/items/{item['id']}", "/api/v1/items?limit=100"):
            code, body = _call("GET", path)
            self.assertEqual(code, 200, body)
            rows = [body] if "items" not in body else body["items"]
            row = next((r for r in rows if r["id"] == item["id"]), None)
            self.assertIsNotNone(row, f"{path} omitted the item")
            for key, empty in (("features", []), ("dimensions", {}), ("extra", {})):
                self.assertEqual(row[key], empty,
                                 f"{path}: {key} must be {empty!r}, got {row[key]!r}")
            self.assertEqual(row["assets"], [])

    def test_list_items_returns_assets_and_counts(self):
        """Regression: the batched assets/counts query 500'd on every list.

        pgx sends a Go []string as text[], and `uuid = ANY(text[])` is a type
        error. The create/read paths never touched the list query, so only this
        assertion caught it — hence it lives here, in the live-stack suite.
        """
        item = make_item()
        png = tiny_png()
        body, headers = multipart_file("file", "b.png", png)
        code, _ = _call("POST", f"/api/v1/items/{item['id']}/assets",
                        raw=body, headers=headers)
        self.assertEqual(code, 201)

        code, page = _call("GET", "/api/v1/items?limit=100")
        self.assertEqual(code, 200, page)
        self.assertIn("items", page)
        row = next((i for i in page["items"] if i["id"] == item["id"]), None)
        self.assertIsNotNone(row, f"created item missing from the list: {page}")
        # the batched joins must actually populate, not silently return empty
        self.assertEqual(len(row["assets"]), 1)
        self.assertEqual(row["assets"][0]["width"], 8)
        for key in ("chunks", "embeddings_text", "embeddings_image"):
            self.assertIn(key, row["counts"])
            self.assertIsInstance(row["counts"][key], int)

    def test_list_items_paginates(self):
        ids = {make_item()["id"] for _ in range(3)}
        seen, cursor = set(), ""
        for _ in range(10):
            path = "/api/v1/items?limit=2" + (f"&cursor={cursor}" if cursor else "")
            code, page = _call("GET", path)
            self.assertEqual(code, 200, page)
            seen.update(i["id"] for i in page["items"])
            cursor = page.get("next_cursor")
            if not cursor:
                break
        self.assertTrue(ids.issubset(seen), f"paging lost items: {ids - seen}")

    def test_list_items_filters(self):
        cat = f"probe-{uuid.uuid4().hex[:6]}"
        code, _ = _call("POST", "/api/v1/items", body={
            "sku": f"LIST-{cat}", "title": "Listing Probe", "category": cat,
            "status": "draft"})
        self.assertEqual(code, 201)
        code, page = _call("GET", f"/api/v1/items?category={cat}")
        self.assertEqual(code, 200, page)
        self.assertEqual([i["category"] for i in page["items"]], [cat])
        code, page = _call("GET", "/api/v1/items?query=Listing+Probe")
        self.assertEqual(code, 200, page)
        self.assertTrue(any(i["id"] for i in page["items"]))

    def test_list_generation_jobs_returns_nested_rows(self):
        """The batched job query must assemble drafts/reviews per job."""
        code, page = _call("GET", "/api/v1/descriptions/jobs?limit=20")
        self.assertEqual(code, 200, page)
        for job in page["items"]:
            for key in ("drafts", "reviews", "published"):
                self.assertIsInstance(job.get(key), list)
            self.assertEqual(job["drafts"], sorted(job["drafts"],
                                                   key=lambda d: d["rank"]))

    def test_asset_upload(self):
        item = make_item()
        png = tiny_png()
        body, headers = multipart_file("file", "bottle.png", png)
        code, asset = _call("POST", f"/api/v1/items/{item['id']}/assets",
                            raw=body, headers=headers)
        self.assertEqual(code, 201, asset)
        self.assertEqual(asset["mime"], "image/png")
        self.assertEqual(asset["width"], 8)
        self.assertEqual(asset["sha256"], __import__("hashlib").sha256(png).hexdigest())


class TestProject1QA(unittest.TestCase):
    def test_ingest_and_grounded_answer(self):
        item = make_item()
        code, _ = _call("PATCH", f"/api/v1/items/{item['id']}",
                        body={"status": "active"})
        self.assertEqual(code, 200)
        body, headers = multipart_file("file", "bottle.png", tiny_png())
        code, _ = _call("POST", f"/api/v1/items/{item['id']}/assets",
                        raw=body, headers=headers)
        self.assertEqual(code, 201)

        code, job = _call("POST", f"/api/v1/items/{item['id']}/ingest", body={})
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/jobs/ingest/{job['job_id']}")
        self.assertEqual(done["status"], "succeeded", done)
        self.assertGreater(done["chunks_indexed"], 0)

        code, res = _call("POST", "/api/v1/qa/ask", body={
            "question": QUESTION,
            "top_k": 6, "use_images": True, "composition": "extractive",
            "user_ref": "itest:buyer",
        })
        self.assertEqual(code, 200, res)
        self.assertIn("answer", res)
        self.assertIn("citation_check", res)
        self.assertIsInstance(res["citation_check"]["passed"], bool)
        self.assertIn("citations", res)
        self.assertIn("retrieval", res)
        self.assertIn("latency_ms", res)
        # contract rule: every citation carries sentence_index + modality + score
        for cit in res["citations"]:
            self.assertIn("sentence_index", cit)
            self.assertIn(cit["modality"], ("text", "image"))
            self.assertIsInstance(cit["score"], (int, float))

        # history + fetch round-trip
        code, hist = _call("GET", "/api/v1/qa/history?user_ref=itest:buyer")
        self.assertEqual(code, 200)
        row = next((a for a in hist.get("items", [])
                    if a.get("answer_id") == res["answer_id"]), None)
        self.assertIsNotNone(row, f"answer not in history: {hist}")

        # These three were each a real bug: the history table rendered blank
        # because the join and the JSON tag were both missing.
        self.assertTrue(row.get("question"), "history must carry the question")
        self.assertEqual(row.get("user_ref"), "itest:buyer")
        self.assertIsInstance(row.get("citation_check_passed"), bool)

        code, stored = _call("GET", f"/api/v1/qa/answers/{res['answer_id']}")
        self.assertEqual(code, 200)
        self.assertEqual(stored["answer"], res["answer"])
        self.assertIsInstance(stored.get("citation_check_passed"), bool)
        self.assertEqual(stored.get("question"), QUESTION,
                         "the stored answer must carry the question it answered")


class TestProject2Descriptions(unittest.TestCase):
    def test_editor_gate_flow(self):
        # A publishable spec must reference a catalogue item: publish resolves
        # the target item from spec.item_id (or the request body) and returns
        # 422 without one. Creating the item first mirrors the real flow.
        item = make_item()
        code, job = _call("POST", "/api/v1/descriptions/jobs", key=EDITOR_KEY, body={
            "spec": {
                "item_id": item["id"],
                "title": "Kestrel 12 oz Insulated Bottle",
                "category": "drinkware",
                "material": "18/8 stainless steel",
                "dimensions": {"height_cm": 26.0, "capacity_oz": 12},
                "features": ["double-wall vacuum insulation", "leak-proof lid"],
            },
            "beam_width": 4, "max_len": 192,
        })
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/descriptions/jobs/{job['job_id']}")
        self.assertEqual(done["status"], "draft_ready", done)
        self.assertTrue(done.get("drafts"), "no drafts returned")
        self.assertEqual(done["drafts"][0]["rank"], 1)

        # publish before approval must FAIL (editor gate)
        code, early = _call("POST",
                            f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                            key=EDITOR_KEY, body={})
        self.assertIn(code, (409, 422), early)

        # edit-inline path: editor rewrites, then publish
        top = done["drafts"][0]
        code, reviewed = _call(
            "POST", f"/api/v1/descriptions/jobs/{job['job_id']}/review",
            key=EDITOR_KEY,
            body={"decision": "edit", "draft_id": top["id"],
                  "edited_text": top["text"] + " Built for the daily commute.",
                  "notes": "tightened the close"})
        self.assertEqual(code, 200, reviewed)
        self.assertEqual(reviewed["status"], "approved")

        # Reviewing again must fail. The review row and the status change commit
        # as one transaction, so the second call sees `approved` (not
        # `draft_ready`) and the CanReview guard rejects it. Two writes left a
        # duplicate-approval window open.
        code, again = _call(
            "POST", f"/api/v1/descriptions/jobs/{job['job_id']}/review",
            key=EDITOR_KEY, body={"decision": "approve", "draft_id": top["id"]})
        self.assertIn(code, (409, 422), again)

        code, pub = _call("POST",
                          f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                          key=EDITOR_KEY, body={})
        self.assertEqual(code, 201, pub)
        self.assertTrue(pub["text"].endswith("Built for the daily commute."),
                        f"published text must be the editor's: {pub['text']!r}")

        # published row records who approved it and which draft it came from
        code, rows = _call("GET", f"/api/v1/descriptions/published/{pub['item_id']}")
        self.assertEqual(code, 200)
        self.assertTrue(rows.get("items"))
        self.assertNotEqual(rows["items"][0].get("approved_by"), "editor",
                            "approver must be the authenticated key, not a literal")
        self.assertTrue(rows["items"][0].get("draft_id"),
                        "published_descriptions.draft_id must be recorded")

    def test_publish_without_an_item_is_refused(self):
        """422, not a silent publish against nothing."""
        code, job = _call("POST", "/api/v1/descriptions/jobs", key=EDITOR_KEY, body={
            "spec": {"title": "Orphan Spec", "category": "home"},
            "beam_width": 2, "max_len": 64})
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/descriptions/jobs/{job['job_id']}")
        self.assertEqual(done["status"], "draft_ready", done)
        code, body = _call("POST", f"/api/v1/descriptions/jobs/{job['job_id']}/review",
                           key=EDITOR_KEY,
                           body={"decision": "approve", "draft_id": done["drafts"][0]["id"]})
        self.assertEqual(code, 200, body)
        code, body = _call("POST", f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                           key=EDITOR_KEY, body={})
        self.assertEqual(code, 422, body)

    def test_publishing_never_falls_back_to_a_raw_draft(self):
        """The gate bypass this replaces.

        The publish handler used to fall back to `drafts[0].text` — the raw
        model output — whenever no review matched. If a job could reach
        `approved` without an approval row, that path published unreviewed
        generation output. Approval is only reachable through /review, so the
        live assertion is that a job with drafts but no review is refused.
        """
        lamp = make_item()
        code, job = _call("POST", "/api/v1/descriptions/jobs", key=EDITOR_KEY, body={
            "spec": {"item_id": lamp["id"], "title": "Atlas Table Lamp",
                     "category": "home", "material": "borosilicate glass",
                     "features": ["dimmable"]},
            "beam_width": 2, "max_len": 96})
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/descriptions/jobs/{job['job_id']}")
        self.assertEqual(done["status"], "draft_ready", done)
        self.assertTrue(done.get("drafts"))

        # still draft_ready -> refused
        code, _ = _call("POST", f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                        key=EDITOR_KEY, body={})
        self.assertIn(code, (409, 422))

        # an approval with a draft_id that does not belong to the job is refused
        code, _ = _call(
            "POST", f"/api/v1/descriptions/jobs/{job['job_id']}/review",
            key=EDITOR_KEY,
            body={"decision": "approve", "draft_id": "00000000-0000-0000-0000-000000000000"})
        self.assertIn(code, (409, 422))

        # and the job is still not publishable
        code, _ = _call("POST", f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                        key=EDITOR_KEY, body={})
        self.assertIn(code, (409, 422))

    def test_reject_path(self):
        code, job = _call("POST", "/api/v1/descriptions/jobs", key=EDITOR_KEY, body={
            "spec": {"title": "Ridge 40L Travel Pack", "category": "bags",
                     "material": "recycled nylon",
                     "features": ["laptop sleeve", "rain cover"]},
            "beam_width": 2, "max_len": 128,
        })
        self.assertEqual(code, 202, job)
        done = wait_job(f"/api/v1/descriptions/jobs/{job['job_id']}")
        self.assertEqual(done["status"], "draft_ready", done)
        code, rejected = _call(
            "POST", f"/api/v1/descriptions/jobs/{job['job_id']}/review",
            key=EDITOR_KEY,
            body={"decision": "reject", "notes": "tone off-brand"})
        self.assertEqual(code, 200, rejected)
        self.assertEqual(rejected["status"], "rejected")
        code, _ = _call("POST",
                        f"/api/v1/descriptions/jobs/{job['job_id']}/publish",
                        key=EDITOR_KEY, body={})
        self.assertIn(code, (409, 422))


if __name__ == "__main__":
    unittest.main(verbosity=2)
    sys.exit(0)
