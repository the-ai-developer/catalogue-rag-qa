package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	reqTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "api_requests_total", Help: "HTTP requests by route and status"},
		[]string{"route", "status"})
	reqDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "api_request_duration_seconds",
		Help:    "HTTP latency by route",
		Buckets: prometheus.DefBuckets,
	}, []string{"route"})
)

func requestID(next http.Handler) http.Handler {
	return http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" || len(id) > 128 {
			buf := make([]byte, 8)
			_, _ = rand.Read(buf)
			id = hex.EncodeToString(buf)
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	}))
}

// routePattern is the matched chi route template ("/api/v1/items/{id}"), or
// "unmatched" for 404s. Using r.URL.Path here would mint a new time series per
// /items/{uuid} and eventually take the metrics backend down with it.
func routePattern(r *http.Request) string {
	if rctx := chi.RouteContext(r.Context()); rctx != nil {
		if p := rctx.RoutePattern(); p != "" {
			return p
		}
	}
	if r.URL.Path == "" {
		return "/"
	}
	return "unmatched"
}

func slogMiddleware(next http.Handler) http.Handler {
	return http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		route := routePattern(r)
		reqTotal.WithLabelValues(route, statusText(sw.status)).Inc()
		reqDuration.WithLabelValues(route).Observe(time.Since(start).Seconds())
		slog.Info("http", "method", r.Method, "path", route,
			"status", sw.status, "duration_ms", time.Since(start).Milliseconds(),
			"request_id", r.Context().Value(requestIDKey))
	}))
}

func statusText(code int) string {
	if code == 0 {
		return "unknown"
	}
	return http.StatusText(code)
}

// statusWriter records the status code and, importantly, reports 204 for
// handlers that write nothing (DELETE /items/{id}) instead of claiming 200.
type statusWriter struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (w *statusWriter) WriteHeader(code int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(b)
}

// Flush lets streaming handlers (SSE) keep working through the wrapper.
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func recoverer(next http.Handler) http.Handler {
	return http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				slog.Error("panic recovered", "err", rec, "path", routePattern(r),
					"request_id", r.Context().Value(requestIDKey))
				// A panic after the header was written cannot be turned into a
				// 500 any more; log loudly rather than corrupting the response.
				if sw, ok := w.(*statusWriter); ok && sw.wroteHeader {
					return
				}
				writeJSON(w, http.StatusInternalServerError, map[string]any{
					"error": map[string]any{"code": "internal",
						"message": "internal error", "details": map[string]any{}}})
			}
		}()
		next.ServeHTTP(w, r)
	}))
}

func strPtr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}
