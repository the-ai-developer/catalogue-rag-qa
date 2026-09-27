package httpapi

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"catalogue-ai/services/api/internal/apperr"
	"catalogue-ai/services/api/internal/config"
)

func TestHashKey(t *testing.T) {
	if HashKey("secret") != HashKey("secret") {
		t.Fatal("hash must be deterministic")
	}
	if HashKey("secret") == HashKey("other") {
		t.Fatal("different keys must hash differently")
	}
	// The documented dev key must keep hashing to the value shipped in
	// .env.example / docker-compose, or every scripted demo silently 401s.
	const wantDev = "6f7593f48eb57fc679c0b173ee15f27ea20f32b92d146e414327218a079ab479"
	if got := HashKey("local-dev-admin-key"); got != wantDev {
		t.Fatalf("local-dev-admin-key hash = %s, want %s", got, wantDev)
	}
}

func TestRequireRoles(t *testing.T) {
	keys := staticKeys{
		HashKey("viewer-key"): {Name: "v", Role: "viewer"},
		HashKey("editor-key"): {Name: "e", Role: "editor"},
		HashKey("admin-key"):  {Name: "a", Role: "admin"},
	}
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(200)
	})

	cases := []struct {
		name, key, needed string
		want              int
	}{
		{"no key", "", "viewer", 401},
		{"unknown key", "bogus", "viewer", 401},
		{"viewer reads", "viewer-key", "viewer", 200},
		{"viewer blocked from admin", "viewer-key", "admin", 403},
		{"editor on editor route", "editor-key", "editor", 200},
		{"editor blocked from admin", "editor-key", "admin", 403},
		{"admin does everything", "admin-key", "admin", 200},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/api/v1/items", nil)
			if c.key != "" {
				req.Header.Set("X-API-Key", c.key)
			}
			rec := httptest.NewRecorder()
			Require(keys, c.needed)(ok).ServeHTTP(rec, req)
			if rec.Code != c.want {
				t.Fatalf("status=%d want %d body=%s", rec.Code, c.want, rec.Body.String())
			}
		})
	}
}

func TestBootstrapKeysFallback(t *testing.T) {
	cfg := &config.Config{BootstrapAPIKeys: []config.BootstrapKey{
		{Name: "local", KeyHash: HashKey("boot"), Role: "admin"}}}
	chain := Chain{staticKeys{}, BootstrapKeys(cfg)}
	role, err := chain.RoleForKeyHash(context.Background(), HashKey("boot"))
	if err != nil || role != "admin" {
		t.Fatalf("bootstrap key not resolved: %s %v", role, err)
	}
	role, err = chain.RoleForKeyHash(context.Background(), "nope")
	if err != nil || role != "" {
		t.Fatalf("unknown key must not resolve: %q %v", role, err)
	}
}

// errVerifier models a key store that is down.
type errVerifier struct{ err error }

func (e errVerifier) RoleForKeyHash(context.Context, string) (string, error) {
	return "", e.err
}

func TestChainDoesNotFallBackOnInfraError(t *testing.T) {
	// A database outage must not resolve keys through the bootstrap verifier:
	// that would silently widen access to every bootstrap admin key.
	boot := BootstrapKeys(&config.Config{BootstrapAPIKeys: []config.BootstrapKey{
		{Name: "local", KeyHash: HashKey("boot"), Role: "admin"}}})
	chain := Chain{errVerifier{errors.New("connection refused")}, boot}
	role, err := chain.RoleForKeyHash(context.Background(), HashKey("boot"))
	if err == nil {
		t.Fatal("expected the infra error to propagate")
	}
	if role != "" {
		t.Fatalf("role must be empty on error, got %q", role)
	}
}

func TestRequireReturns500Not401WhenKeyStoreIsDown(t *testing.T) {
	// The original bug: any DB error became "unknown API key" -> 401, so a
	// database blip looked like a bad credential and logged everyone out.
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(200)
	})
	h := Require(Chain{errVerifier{apperr.ErrAuthUnavailable}}, "viewer")(ok)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/items", nil)
	req.Header.Set("X-API-Key", "anything")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code == http.StatusUnauthorized {
		t.Fatalf("key-store outage must not be reported as 401, got %d: %s",
			rec.Code, rec.Body.String())
	}
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
}

func TestRequireAttachesAuthenticatedPrincipal(t *testing.T) {
	keys := namedKeys{HashKey("k"): {KeyName: "ci-bot", Role: "editor"}}
	var got Principal
	var found bool
	h := Require(keys, "viewer")(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, found = FromContext(r.Context())
		w.WriteHeader(200)
	}))
	req := httptest.NewRequest(http.MethodGet, "/api/v1/items", nil)
	req.Header.Set("X-API-Key", "k")
	// A caller trying to attribute the action to somebody else.
	req.Header.Set("X-Api-Key-Name", "chief-editor")
	h.ServeHTTP(httptest.NewRecorder(), req)

	if !found {
		t.Fatal("principal must be on the request context")
	}
	if got.KeyName != "ci-bot" {
		t.Fatalf("KeyName = %q, want the authenticated name (header must be ignored)", got.KeyName)
	}
	if got.Role != "editor" {
		t.Fatalf("Role = %q", got.Role)
	}
}

type namedKeys map[string]Principal

func (n namedKeys) RoleForKeyHash(_ context.Context, hash string) (string, error) {
	if p, ok := n[hash]; ok {
		return p.Role, nil
	}
	return "", nil
}

func (n namedKeys) NameForKeyHash(hash string) (string, bool) {
	p, ok := n[hash]
	return p.KeyName, ok
}

func TestPrincipalNameIgnoresSpoofedHeader(t *testing.T) {
	keys := namedKeys{HashKey("k"): {KeyName: "real", Role: "admin"}}
	var got string
	h := Require(keys, "viewer")(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = principalName(r)
	}))
	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("X-API-Key", "k")
	req.Header.Set("X-Api-Key-Name", "attacker")
	h.ServeHTTP(httptest.NewRecorder(), req)
	if got != "real" {
		t.Fatalf("principalName = %q, want %q", got, "real")
	}
}

func TestUnauthenticatedRouteHasNoPrincipal(t *testing.T) {
	if _, ok := FromContext(context.Background()); ok {
		t.Fatal("background context must carry no principal")
	}
}

type fakeStore struct {
	role   string
	calls  int
	failAt int
}

func (f *fakeStore) RoleForKeyHash(_ context.Context, _ string) (string, error) {
	f.calls++
	if f.failAt > 0 && f.calls >= f.failAt {
		return "", apperr.ErrAuthUnavailable
	}
	return f.role, nil
}

func TestCachedVerifierCachesThenPropagates(t *testing.T) {
	inner := &fakeStore{role: "viewer"}
	c := NewCached(inner, time.Minute)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		role, err := c.RoleForKeyHash(ctx, "h")
		if err != nil || role != "viewer" {
			t.Fatalf("call %d: role=%q err=%v", i, role, err)
		}
	}
	if inner.calls != 1 {
		t.Fatalf("expected 1 inner call for 5 requests, got %d", inner.calls)
	}

	c.Invalidate("h")
	if _, err := c.RoleForKeyHash(ctx, "h"); err != nil {
		t.Fatal(err)
	}
	if inner.calls != 2 {
		t.Fatalf("invalidate must force a refresh, calls=%d", inner.calls)
	}
}

func TestCachedDoesNotCacheErrors(t *testing.T) {
	inner := &fakeStore{role: "viewer", failAt: 1}
	c := NewCached(inner, time.Minute)
	if _, err := c.RoleForKeyHash(context.Background(), "h"); err == nil {
		t.Fatal("expected the error to surface")
	}
	if _, err := c.RoleForKeyHash(context.Background(), "h"); err == nil {
		t.Fatal("second call must retry, not serve a cached failure")
	}
	if inner.calls != 2 {
		t.Fatalf("inner calls = %d, want 2", inner.calls)
	}
}

func TestCachedDoesNotCacheUnknownKeys(t *testing.T) {
	inner := &fakeStore{role: ""}
	c := NewCached(inner, time.Minute)
	if _, err := c.RoleForKeyHash(context.Background(), "h"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.RoleForKeyHash(context.Background(), "h"); err != nil {
		t.Fatal(err)
	}
	if inner.calls != 2 {
		t.Fatalf("a miss must not be cached, calls=%d", inner.calls)
	}
}

// The bootstrap key's configured name must reach the audit trail. Losing it
// meant every local action was attributed to "key:dfb94e97" instead of
// "local-admin".
func TestBootstrapKeyNameReachesTheContext(t *testing.T) {
	cfg := &config.Config{BootstrapAPIKeys: []config.BootstrapKey{
		{Name: "local-admin", KeyHash: HashKey("secret"), Role: "admin"}}}
	keys := BootstrapKeys(cfg)

	var got Principal
	var ok bool
	h := Require(keys, "viewer")(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, ok = FromContext(r.Context())
		w.WriteHeader(200)
	}))
	req := httptest.NewRequest(http.MethodGet, "/api/v1/items", nil)
	req.Header.Set("X-API-Key", "secret")
	h.ServeHTTP(httptest.NewRecorder(), req)

	if !ok {
		t.Fatal("no principal on the context")
	}
	if got.KeyName != "local-admin" {
		t.Errorf("KeyName = %q, want local-admin", got.KeyName)
	}
	if got.Role != "admin" {
		t.Errorf("Role = %q", got.Role)
	}
}

func TestUnknownKeyFallsBackToAStableNonReversibleTag(t *testing.T) {
	// A verifier that knows roles but not names still has to attribute the
	// action, and must not leak the secret itself.
	keys := namedKeys{HashKey("k"): {KeyName: "", Role: "viewer"}}
	var got string
	h := Require(keys, "viewer")(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = principalName(r)
	}))
	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.Header.Set("X-API-Key", "k")
	h.ServeHTTP(httptest.NewRecorder(), req)
	if !strings.HasPrefix(got, "key:") {
		t.Errorf("principalName = %q, want a key: tag", got)
	}
	// The tag must be a hash of the key, never the key itself.
	if got == "k" || strings.Contains(strings.TrimPrefix(got, "key:"), "k") {
		t.Errorf("the fallback tag must not embed the raw key: %q", got)
	}
}
