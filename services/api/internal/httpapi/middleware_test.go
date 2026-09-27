package httpapi

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/go-chi/chi/v5"
)

func TestRoutePatternIsTheTemplateNotThePath(t *testing.T) {
	// Using r.URL.Path as a Prometheus label minted a new time series per
	// /items/{uuid} and would eventually take the metrics backend down.
	var label string
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		label = routePattern(r)
		w.WriteHeader(http.StatusOK)
	})

	router := chi.NewRouter()
	router.Use(slogMiddleware)
	router.Get("/api/v1/items/{id}", inner)

	// Two different ids must produce one identical label.
	for _, id := range []string{"6f1a-1111", "9b2c-2222", "0000-dead"} {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/v1/items/"+id, nil))
		if label != "/api/v1/items/{id}" {
			t.Fatalf("routePattern = %q, want the template", label)
		}
	}
}

func TestRoutePatternFallsBackWhenUnmatched(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/nope", nil)
	if got := routePattern(req); got != "unmatched" {
		t.Errorf("routePattern = %q, want unmatched", got)
	}
}

func TestStatusWriterReportsNoContent(t *testing.T) {
	// A handler that writes nothing used to be logged as 200 while the client
	// received 204, so the metric for DELETEs was always wrong.
	rec := httptest.NewRecorder()
	w := &statusWriter{ResponseWriter: rec, status: http.StatusOK}
	handlerThatWritesNothing(w)
	if w.status != http.StatusNoContent {
		t.Errorf("status = %d, want 204", w.status)
	}
}

func handlerThatWritesNothing(w http.ResponseWriter) { w.WriteHeader(http.StatusNoContent) }

func TestStatusWriterIgnoresASecondWriteHeader(t *testing.T) {
	rec := httptest.NewRecorder()
	w := &statusWriter{ResponseWriter: rec, status: http.StatusOK}
	w.WriteHeader(http.StatusCreated)
	w.WriteHeader(http.StatusTeapot)
	if w.status != http.StatusCreated {
		t.Errorf("status = %d, want the first code 201", w.status)
	}
}

func TestStatusWriterInfersOKFromWrite(t *testing.T) {
	rec := httptest.NewRecorder()
	w := &statusWriter{ResponseWriter: rec, status: http.StatusOK}
	_, _ = w.Write([]byte("hi"))
	if w.status != http.StatusOK {
		t.Errorf("status = %d, want 200", w.status)
	}
}

func TestRecovererTurnsAPanicInto500(t *testing.T) {
	h := recoverer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		panic("boom")
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/x", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", rec.Code)
	}
}

func TestRecovererDoesNotDoubleWriteAfterHeadersAreSent(t *testing.T) {
	h := recoverer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("partial"))
		panic("late boom")
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/x", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want the already-sent 200", rec.Code)
	}
}

func TestRequestIDPassesThroughAndGenerates(t *testing.T) {
	var seen string
	h := requestID(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		seen, _ = r.Context().Value(requestIDKey).(string)
	}))

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/x", nil))
	if seen == "" || rec.Header().Get("X-Request-ID") != seen {
		t.Errorf("generated id not propagated: ctx=%q header=%q", seen,
			rec.Header().Get("X-Request-ID"))
	}

	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("X-Request-ID", "trace-123")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Header().Get("X-Request-ID") != "trace-123" {
		t.Errorf("client request id was not echoed")
	}

	// An unbounded client-supplied id is a log-injection / memory vector.
	req = httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("X-Request-ID", string(make([]byte, 4096)))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if len(rec.Header().Get("X-Request-ID")) > 128 {
		t.Errorf("an oversized X-Request-ID must be replaced, got %d bytes",
			len(rec.Header().Get("X-Request-ID")))
	}
}

func TestStatusTextNeverEmpty(t *testing.T) {
	if statusText(0) != "unknown" {
		t.Error("code 0 must map to a non-empty label")
	}
	if statusText(http.StatusOK) != "OK" {
		t.Error("200 must map to OK")
	}
}
