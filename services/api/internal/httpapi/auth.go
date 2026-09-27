package httpapi

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"catalogue-ai/services/api/internal/apperr"
	"catalogue-ai/services/api/internal/config"
)

// Role ranks: viewer < editor < admin.
var roleRank = map[string]int{"viewer": 1, "editor": 2, "admin": 3}

// ErrAuthUnavailable means the key store could not be consulted (database down,
// pool exhausted, timeout). It is deliberately distinct from "key not found":
// collapsing the two turns a database blip into a 401 and signs every user out
// while looking like a credential problem.
var ErrAuthUnavailable = apperr.ErrAuthUnavailable

// KeyVerifier resolves an API key hash to a role.
type KeyVerifier interface {
	RoleForKeyHash(ctx context.Context, hash string) (string, error)
}

// namedRole is a resolved key: its configured name and its role.
type namedRole struct {
	Name string
	Role string
}

// staticKeys holds the bootstrap keys. The *name* is kept so audit rows and
// review records can attribute an action to a person or a bot; without it every
// bootstrap action landed as an opaque hash.
type staticKeys map[string]namedRole // sha256hex -> {name, role}

func (s staticKeys) RoleForKeyHash(_ context.Context, hash string) (string, error) {
	entry, ok := s[hash]
	if !ok {
		return "", nil
	}
	return entry.Role, nil
}

// NameForKeyHash satisfies namer for bootstrap keys.
func (s staticKeys) NameForKeyHash(hash string) (string, bool) {
	entry, ok := s[hash]
	if !ok || entry.Name == "" {
		return "", false
	}
	return entry.Name, true
}

// BootstrapKeys turns BOOTSTRAP_API_KEYS entries (name:sha256hex:role) into a
// verifier used before/alongside the api_keys table.
func BootstrapKeys(cfg *config.Config) staticKeys {
	out := staticKeys{}
	for _, k := range cfg.BootstrapAPIKeys {
		out[k.KeyHash] = namedRole{Name: k.Name, Role: k.Role}
	}
	return out
}

// HashKey returns the sha256 hex digest stored for an API key.
func HashKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// Cached memoises a KeyVerifier for a short TTL. The DB-backed verifier runs on
// every authenticated request; without this, API traffic is a per-request
// database round trip on the hot path.
type Cached struct {
	Inner KeyVerifier
	TTL   time.Duration
	Now   func() time.Time

	cache map[string]entry
}

type entry struct {
	role    string
	expires time.Time
}

// NewCached wraps inner with a TTL. A zero or negative TTL disables caching.
func NewCached(inner KeyVerifier, ttl time.Duration) *Cached {
	return &Cached{Inner: inner, TTL: ttl, Now: time.Now, cache: map[string]entry{}}
}

func (c *Cached) RoleForKeyHash(ctx context.Context, hash string) (string, error) {
	now := c.Now()
	if role, ok := c.cache[hash]; ok && now.Before(role.expires) {
		return role.role, nil
	}
	role, err := c.Inner.RoleForKeyHash(ctx, hash)
	if err != nil {
		return "", err
	}
	if role != "" && c.TTL > 0 {
		c.cache[hash] = entry{role: role, expires: now.Add(c.TTL)}
	}
	return role, nil
}

// Invalidate drops a cached role (call on key rotation / deactivation).
func (c *Cached) Invalidate(hash string) { delete(c.cache, hash) }

// NameForKeyHash delegates to the inner verifier when it knows names.
func (c *Cached) NameForKeyHash(hash string) (string, bool) {
	n, ok := c.Inner.(namer)
	if !ok {
		return "", false
	}
	return n.NameForKeyHash(hash)
}

// Chain tries verifiers in order (DB first, then bootstrap). An infrastructure
// error short-circuits: falling through to the bootstrap keys during a database
// outage would silently widen access.
type Chain []KeyVerifier

func (c Chain) RoleForKeyHash(ctx context.Context, hash string) (string, error) {
	for _, v := range c {
		role, err := v.RoleForKeyHash(ctx, hash)
		if err != nil {
			return "", err
		}
		if role != "" {
			return role, nil
		}
	}
	return "", nil
}

// NameForKeyHash asks each verifier in turn for the key's name.
func (c Chain) NameForKeyHash(hash string) (string, bool) {
	for _, v := range c {
		if n, ok := v.(namer); ok {
			if name, found := n.NameForKeyHash(hash); found {
				return name, true
			}
		}
	}
	return "", false
}

type ctxKey string

const (
	// requestIDKey correlates logs, responses and audit rows.
	requestIDKey ctxKey = "request_id"
	// principalKey holds the authenticated caller's key name and role.
	principalKey ctxKey = "principal"
)

// Principal is the authenticated identity attached to a request context.
type Principal struct {
	KeyName string
	Role    string
}

// FromContext returns the authenticated principal. ok is false for routes that
// are deliberately unauthenticated (health, metrics).
func FromContext(ctx context.Context) (Principal, bool) {
	p, ok := ctx.Value(principalKey).(Principal)
	return p, ok
}

// Require returns middleware enforcing at least `needed` role.
func Require(v KeyVerifier, needed string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			key := r.Header.Get("X-API-Key")
			if key == "" {
				writeErr(w, apperr.UnauthorizedErr("missing X-API-Key"))
				return
			}
			role, err := v.RoleForKeyHash(r.Context(), HashKey(key))
			switch {
			case errors.Is(err, ErrAuthUnavailable):
				// Infrastructure problem, not an auth decision: 503 + log, never 401.
				slog.Error("key store unavailable", "err", err, "path", r.URL.Path)
				writeErr(w, apperr.Wrap(apperr.Internal, "auth backend unavailable", err))
				return
			case err != nil:
				slog.Error("key lookup failed", "err", err)
				writeErr(w, apperr.InternalErr("auth lookup failed", err))
				return
			case role == "":
				writeErr(w, apperr.UnauthorizedErr("unknown API key"))
				return
			}
			if roleRank[role] < roleRank[needed] {
				writeErr(w, apperr.ForbiddenErr("role "+role+" < "+needed))
				return
			}
			// The key name comes from the verifier, never from a request header,
			// so audit rows cannot be attributed to somebody else.
			ctx := context.WithValue(r.Context(), principalKey,
				Principal{KeyName: keyNameFor(v, key), Role: role})
			next.ServeHTTP(w, r.WithContext(ctx))
		}))
	}
}

// namer is implemented by verifiers that know the human-readable key name.
type namer interface {
	NameForKeyHash(hash string) (string, bool)
}

// keyNameFor resolves a human-readable name for the key. It never returns an
// empty string: an unattributable audit row is worse than a hashed tag, and a
// verifier that claims to know the name but returns "" must not win.
func keyNameFor(v KeyVerifier, key string) string {
	if n, ok := v.(namer); ok {
		if name, found := n.NameForKeyHash(HashKey(key)); found && name != "" {
			return name
		}
	}
	sum := sha256.Sum256([]byte("name:" + key))
	return "key:" + hex.EncodeToString(sum[:4])
}
