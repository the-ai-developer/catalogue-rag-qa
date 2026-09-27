package config

import (
	"os"
	"strings"
	"testing"
	"time"
)

func withEnv(t *testing.T, kv map[string]string) {
	t.Helper()
	for k, v := range kv {
		t.Setenv(k, v)
	}
}

func TestParseBootstrapKeys(t *testing.T) {
	keys, err := parseBootstrapKeys("a:" + strings.Repeat("1", 64) + ":admin, b:" +
		strings.Repeat("2", 64) + ":VIEWER")
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 2 {
		t.Fatalf("got %d keys, want 2", len(keys))
	}
	if keys[0].Name != "a" || keys[0].Role != "admin" {
		t.Errorf("keys[0] = %+v", keys[0])
	}
	// Roles are lower-cased so "ADMIN" and "admin" cannot diverge.
	if keys[1].Role != "viewer" {
		t.Errorf("role = %q, want lower-cased viewer", keys[1].Role)
	}
}

func TestParseBootstrapKeysLowercasesTheHash(t *testing.T) {
	// HashKey emits lowercase hex, so a hand-typed uppercase hash must still
	// resolve rather than silently failing every request.
	upper := strings.ToUpper(strings.Repeat("ab", 32))
	keys, err := parseBootstrapKeys("a:" + upper + ":admin")
	if err != nil {
		t.Fatal(err)
	}
	if keys[0].KeyHash != strings.ToLower(upper) {
		t.Errorf("KeyHash = %q, want lower-cased", keys[0].KeyHash)
	}
}

func TestParseBootstrapKeysRejectsBadEntries(t *testing.T) {
	bad := []string{
		"only-one-field",
		"a:" + strings.Repeat("1", 64),                   // missing role
		"a:" + strings.Repeat("1", 64) + ":superuser",    // unknown role
		":" + strings.Repeat("1", 64) + ":admin",         // empty name
		"a::admin",                                       // empty hash
		"a:" + strings.Repeat("1", 64) + ":admin,broken", // one bad entry poisons the set
	}
	for _, in := range bad {
		if _, err := parseBootstrapKeys(in); err == nil {
			t.Errorf("parseBootstrapKeys(%q) = nil error, want failure", in)
		}
	}
}

func TestParseBootstrapKeysEmptyIsAllowed(t *testing.T) {
	for _, in := range []string{"", "   ", ","} {
		keys, err := parseBootstrapKeys(in)
		if err != nil {
			t.Errorf("parseBootstrapKeys(%q) errored: %v", in, err)
		}
		if len(keys) != 0 {
			t.Errorf("parseBootstrapKeys(%q) = %v, want none", in, keys)
		}
	}
}

func TestGetDurationRejectsNonsense(t *testing.T) {
	withEnv(t, map[string]string{"T": "banana"})
	if _, err := getduration("T", time.Second); err == nil {
		t.Error("expected a parse error")
	}
	withEnv(t, map[string]string{"T": "0s"})
	if _, err := getduration("T", time.Second); err == nil {
		t.Error("a zero duration must be rejected, not silently defaulted")
	}
	withEnv(t, map[string]string{"T": "-5s"})
	if _, err := getduration("T", time.Second); err == nil {
		t.Error("a negative duration must be rejected")
	}
	withEnv(t, map[string]string{"T": "2500ms"})
	d, err := getduration("T", time.Second)
	if err != nil || d != 2500*time.Millisecond {
		t.Errorf("got %v, %v", d, err)
	}
	os.Unsetenv("T")
	if d, err := getduration("T", 7*time.Second); err != nil || d != 7*time.Second {
		t.Errorf("unset must use the default, got %v %v", d, err)
	}
}

func TestGetIntFallsBackOnGarbage(t *testing.T) {
	withEnv(t, map[string]string{"N": "not-a-number"})
	if got := getint("N", 42); got != 42 {
		t.Errorf("got %d, want the default 42", got)
	}
	withEnv(t, map[string]string{"N": "17"})
	if got := getint("N", 42); got != 17 {
		t.Errorf("got %d, want 17", got)
	}
	os.Unsetenv("N")
	if got := getint("N", 42); got != 42 {
		t.Errorf("got %d, want 42", got)
	}
}

func TestLoadValidatesChunkWindow(t *testing.T) {
	withEnv(t, map[string]string{
		"MAX_CHUNK_TOKENS":     "100",
		"CHUNK_OVERLAP_TOKENS": "100",
	})
	if _, err := Load(); err == nil {
		t.Error("overlap >= max must be rejected: it would loop forever in the chunker")
	}
	withEnv(t, map[string]string{
		"MAX_CHUNK_TOKENS":     "0",
		"CHUNK_OVERLAP_TOKENS": "10",
	})
	if _, err := Load(); err == nil {
		t.Error("max_tokens = 0 must be rejected")
	}
}

func TestLoadRejectsPlaceholderSecrets(t *testing.T) {
	// k8s.yaml ships BOOTSTRAP_API_KEYS with a literal placeholder. If that
	// reaches a real cluster, every deployment shares sha256 of that literal.
	withEnv(t, map[string]string{
		"BOOTSTRAP_API_KEYS": "admin:REPLACE_WITH_SHA256_HEX:admin",
	})
	_, err := Load()
	if err == nil {
		t.Fatal("expected Load to refuse the placeholder")
	}
	if !strings.Contains(err.Error(), "placeholder") {
		t.Errorf("error should name the problem, got: %v", err)
	}
}

func TestLoadRejectsPlaceholderDatabaseURL(t *testing.T) {
	withEnv(t, map[string]string{
		"DATABASE_URL": "postgres://catalogue:CHANGE_ME@db/catalogue",
	})
	if _, err := Load(); err == nil {
		t.Fatal("expected Load to refuse the placeholder password")
	}
}

func TestLoadRejectsPlaceholderModelSecret(t *testing.T) {
	withEnv(t, map[string]string{"MODEL_SHARED_SECRET": "REPLACE_WITH_SHA256_HEX"})
	if _, err := Load(); err == nil {
		t.Fatal("expected Load to refuse the placeholder model secret")
	}
}

func TestLoadTrimsAndDefaults(t *testing.T) {
	withEnv(t, map[string]string{
		"MODEL_SERVER_URL":      "http://model:8090///",
		"ASSET_PUBLIC_BASE_URL": "/assets/",
		"MODEL_SHARED_SECRET":   "  s3cret  ",
	})
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelServerURL != "http://model:8090" {
		t.Errorf("ModelServerURL = %q (trailing slashes must be trimmed)", cfg.ModelServerURL)
	}
	if cfg.AssetPublicBaseURL != "/assets" {
		t.Errorf("AssetPublicBaseURL = %q", cfg.AssetPublicBaseURL)
	}
	if cfg.ModelSharedSecret != "s3cret" {
		t.Errorf("ModelSharedSecret = %q (must be trimmed)", cfg.ModelSharedSecret)
	}
}

func TestLoadDefaultsAreUsable(t *testing.T) {
	withEnv(t, map[string]string{
		"DATABASE_URL":        "",
		"BOOTSTRAP_API_KEYS":  "",
		"MODEL_SHARED_SECRET": "",
		"MODEL_TIMEOUT_SHORT": "",
		"MODEL_TIMEOUT_LONG":  "",
		"AUTH_CACHE_TTL":      "",
	})
	cfg, err := Load()
	if err != nil {
		t.Fatalf("an empty environment must still load: %v", err)
	}
	if cfg.ModelTimeoutLong < 30*time.Second {
		t.Errorf("MODEL_TIMEOUT_LONG default %v is too short for T5 generation", cfg.ModelTimeoutLong)
	}
	if cfg.AuthCacheTTL <= 0 {
		t.Errorf("AUTH_CACHE_TTL default = %v", cfg.AuthCacheTTL)
	}
	if cfg.MigrationsDir == "" {
		t.Error("MigrationsDir must have a default")
	}
}
