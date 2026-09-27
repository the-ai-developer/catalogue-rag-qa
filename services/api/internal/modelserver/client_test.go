package modelserver

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func newTestClient(url string) *Client {
	c := New(url, 2*time.Second, 2*time.Second)
	c.retryN = 2
	return c
}

func TestRetriesOn5xxThenSucceeds(t *testing.T) {
	calls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls < 3 {
			w.WriteHeader(503)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"chunks":[{"ordinal":0,"text":"hi","token_count":1}]}`))
	}))
	defer ts.Close()

	chunks, err := newTestClient(ts.URL).ChunkText(context.Background(), "hi", "copy", 10, 2)
	if err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
	if calls != 3 {
		t.Fatalf("expected 3 attempts (2 retries), got %d", calls)
	}
	if len(chunks) != 1 || chunks[0].Text != "hi" {
		t.Fatalf("bad chunks: %+v", chunks)
	}
}

func TestNoRetryOn4xx(t *testing.T) {
	calls := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(400)
		_, _ = w.Write([]byte(`{"error":"bad"}`))
	}))
	defer ts.Close()

	_, err := newTestClient(ts.URL).EmbedText(context.Background(), []string{"x"})
	if err == nil {
		t.Fatal("expected error for 400")
	}
	if calls != 1 {
		t.Fatalf("4xx must not be retried, got %d attempts", calls)
	}
}

func TestSearchRequestShape(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/search" {
			t.Errorf("path=%s", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"hits":[],"models":{"text":"m"}}`))
	}))
	defer ts.Close()

	res, err := newTestClient(ts.URL).Search(context.Background(), "q", nil, 6,
		nil, "any", true)
	if err != nil {
		t.Fatal(err)
	}
	if res.Models["text"] != "m" {
		t.Fatalf("bad response: %+v", res)
	}
}
