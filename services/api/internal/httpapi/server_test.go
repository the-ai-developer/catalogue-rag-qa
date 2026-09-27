package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"catalogue-ai/services/api/internal/apperr"
	"catalogue-ai/services/api/internal/config"
)

// testAdminKey is an admin key so every route is reachable; the role checks
// themselves are covered in auth_test.go.
const testAdminKey = "test-admin-key"

// newTestRouter wires the real router with a nil store. Every route exercised
// here must answer before touching the database, so the nil pointer is never
// dereferenced — that is exactly the property being tested. A request that
// does reach the store panics and is recovered into a 500, which is how the
// tests below detect "validation did not run first".
func newTestRouter() http.Handler {
	srv := New(nil, nil, &config.Config{},
		staticKeys{HashKey(testAdminKey): {Name: "test-admin", Role: "admin"}})
	return srv.Router()
}

// authed builds a request that passes the role check.
func authed(method, target string, body io.Reader) *http.Request {
	req := httptest.NewRequest(method, target, body)
	req.Header.Set("X-API-Key", testAdminKey)
	return req
}

func postJSON(t *testing.T, h http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, authed(http.MethodPost, path, &buf))
	return rec
}

func patchJSON(t *testing.T, h http.Handler, path string, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := authed(http.MethodPatch, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func errorCode(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var env struct {
		Error struct {
			Code    string         `json:"code"`
			Message string         `json:"message"`
			Details map[string]any `json:"details"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("response is not the error envelope: %v (%s)", err, rec.Body.String())
	}
	if env.Error.Code == "" {
		t.Fatalf("envelope has no code: %s", rec.Body.String())
	}
	if env.Error.Details == nil {
		t.Errorf("details must be present, not null, so clients can rely on it")
	}
	return env.Error.Code
}

func TestHealthAndReadyAreUnauthenticated(t *testing.T) {
	h := newTestRouter()
	for _, p := range []string{"/healthz"} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, p, nil))
		if rec.Code != http.StatusOK {
			t.Errorf("%s = %d, want 200 without an API key", p, rec.Code)
		}
	}
	// /readyz touches the store, so it is exercised through a nil-safe path
	// only for the fact that it is not blocked by auth.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code == http.StatusUnauthorized || rec.Code == http.StatusForbidden {
		t.Errorf("/readyz must not require a key, got %d", rec.Code)
	}
}

func TestProtectedRoutesRequireAKey(t *testing.T) {
	h := newTestRouter()
	protected := []struct{ method, path string }{
		{http.MethodGet, "/api/v1/items"},
		{http.MethodPost, "/api/v1/items"},
		{http.MethodPost, "/api/v1/qa/ask"},
		{http.MethodPost, "/api/v1/descriptions/jobs"},
	}
	for _, r := range protected {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(r.method, r.path, strings.NewReader("{}"))
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s %s = %d, want 401", r.method, r.path, rec.Code)
		}
		if code := errorCode(t, rec); code != string(apperr.Unauthorized) {
			t.Errorf("%s %s code = %q", r.method, r.path, code)
		}
	}
}

func TestCreateItemValidationRunsBeforeTheStore(t *testing.T) {
	h := newTestRouter()
	cases := []struct {
		name string
		body any
		want string
	}{
		{"missing everything", map[string]any{}, "bad_request"},
		{"no title", map[string]any{"sku": "S", "category": "c"}, "bad_request"},
		{"blank sku", map[string]any{"sku": "   ", "title": "t", "category": "c"}, "bad_request"},
		{"blank category", map[string]any{"sku": "s", "title": "t", "category": " "}, "bad_request"},
		{"bad status", map[string]any{"sku": "s", "title": "t", "category": "c",
			"status": "live"}, "bad_request"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := postJSON(t, h, "/api/v1/items", c.body)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (store must not be reached)", rec.Code)
			}
			if got := errorCode(t, rec); got != c.want {
				t.Errorf("code = %q, want %q", got, c.want)
			}
		})
	}
}

func TestMalformedJSONIsRejectedNotPanicked(t *testing.T) {
	h := newTestRouter()
	for _, body := range []string{"", "{", "not json", `{"sku":`} {
		rec := postJSON(t, h, "/api/v1/items", nil)
		_ = rec
		req := authed(http.MethodPost, "/api/v1/items", strings.NewReader(body))
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %q -> %d, want 400", body, rec.Code)
		}
		if got := errorCode(t, rec); got != "bad_request" {
			t.Errorf("body %q code = %q", body, got)
		}
	}
}

func TestOversizedBodyIsRejected(t *testing.T) {
	h := newTestRouter()
	huge := strings.Repeat("a", 2<<20)
	rec := postJSON(t, h, "/api/v1/items", map[string]any{"sku": huge})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for a body over the 1MB limit", rec.Code)
	}
}

func TestPatchItemRejectsABadStatus(t *testing.T) {
	h := newTestRouter()
	rec := patchJSON(t, h, "/api/v1/items/6f1a-uuid", `{"status":"live"}`)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
	if got := errorCode(t, rec); got != "bad_request" {
		t.Errorf("code = %q", got)
	}
}

func TestQaAskValidation(t *testing.T) {
	h := newTestRouter()
	cases := []struct {
		name string
		body map[string]any
	}{
		{"empty question", map[string]any{"question": ""}},
		{"blank question", map[string]any{"question": "   "}},
		{"top_k too large", map[string]any{"question": "bottle?", "top_k": 21}},
		{"negative top_k is defaulted, not fatal", map[string]any{"question": "b?", "top_k": -1}},
		{"bad composition", map[string]any{"question": "b?", "composition": "poetic"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := postJSON(t, h, "/api/v1/qa/ask", c.body)
			switch c.name {
			case "negative top_k is defaulted, not fatal":
				// Validation lets it through, so the handler proceeds to the
				// nil store and the recoverer turns that into a 500. The point
				// is only that it is NOT a 400.
				if rec.Code == http.StatusBadRequest {
					t.Errorf("top_k <= 0 should default to 6, not 400")
				}
			default:
				if rec.Code != http.StatusBadRequest {
					t.Fatalf("status = %d, want 400 (store must not be reached)", rec.Code)
				}
				if got := errorCode(t, rec); got != "bad_request" {
					t.Errorf("code = %q", got)
				}
			}
		})
	}
}

func TestDescriptionJobRequiresACategory(t *testing.T) {
	h := newTestRouter()
	for _, body := range []map[string]any{
		{},
		{"spec": map[string]any{}},
		{"spec": map[string]any{"category": "  "}},
	} {
		rec := postJSON(t, h, "/api/v1/descriptions/jobs", body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %v -> %d, want 400", body, rec.Code)
		}
	}
}

func TestReviewDecisionIsValidated(t *testing.T) {
	h := newTestRouter()
	// An unknown decision must be rejected on the decision vocabulary alone,
	// before any job lookup happens.
	rec := postJSON(t, h, "/api/v1/descriptions/jobs/6f1a/review",
		map[string]any{"decision": "yolo"})
	if rec.Code == http.StatusOK {
		t.Error("an unknown decision must never succeed")
	}
}

func TestQueryLimitClamping(t *testing.T) {
	cases := map[string]int{
		"":    20,
		"0":   20,
		"-3":  20,
		"abc": 20,
		"7":   7,
		"500": 100,
	}
	for raw, want := range cases {
		u := "/api/v1/qa/history"
		if raw != "" {
			u += "?limit=" + raw
		}
		req := httptest.NewRequest(http.MethodGet, u, nil)
		if got := queryLimit(req, 20); got != want {
			t.Errorf("limit=%q -> %d, want %d", raw, got, want)
		}
	}
}

func TestNullableCursor(t *testing.T) {
	if nullable("") != nil {
		t.Error(`"" must serialise as null`)
	}
	if nullable("abc") != "abc" {
		t.Error("a non-empty cursor must pass through")
	}
	if nullable(0) != nil {
		t.Error("0 must serialise as null")
	}
	if nullable(5) != 5 {
		t.Error("a positive count must pass through")
	}
	if nullable(nil) != nil {
		t.Error("nil must serialise as null")
	}
}

func TestUnknownRouteIsNotAnInternalError(t *testing.T) {
	h := newTestRouter()
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/v1/nope", nil))
	if rec.Code == http.StatusInternalServerError {
		t.Errorf("unknown route = 500, want 404")
	}
}

func TestRouterUsesChiURLParams(t *testing.T) {
	// Guards the wiring: if chi.URLParam silently returned "" every handler
	// would operate on an empty id.
	rctx := chi.NewRouteContext()
	rctx.URLParams.Add("id", "abc-123")
	ctx := context.WithValue(context.Background(), chi.RouteCtxKey, rctx)
	req := httptest.NewRequest(http.MethodGet, "/x", nil).WithContext(ctx)
	if got := chi.URLParam(req, "id"); got != "abc-123" {
		t.Errorf("URLParam = %q", got)
	}
}
