package httpapi

import (
	"strings"
	"testing"

	"catalogue-ai/services/api/internal/config"
	"catalogue-ai/services/api/internal/store"
)

func assetServer(base string) *Server {
	return New(nil, nil, &config.Config{AssetPublicBaseURL: base}, staticKeys{})
}

// item_assets.storage_path is stored relative to ASSET_STORAGE_DIR. Every asset
// that leaves the API must carry the public prefix, or the browser resolves it
// against the current route and gets the SPA fallback instead of an image.
func TestPublicAssetURLIsPrefixed(t *testing.T) {
	s := assetServer("/assets")
	if got := s.publicAssetURL("42/ab12.png"); got != "/assets/42/ab12.png" {
		t.Errorf("got %q, want /assets/42/ab12.png", got)
	}
}

func TestPublicAssetURLIsIdempotent(t *testing.T) {
	// Applying the helper twice (or to an already-prefixed value) must not
	// produce /assets/assets/....
	s := assetServer("/assets")
	once := s.publicAssetURL("42/ab12.png")
	twice := s.publicAssetURL(once)
	if once != twice {
		t.Errorf("not idempotent: %q -> %q", once, twice)
	}
}

func TestPublicAssetURLLeavesAbsoluteValuesAlone(t *testing.T) {
	s := assetServer("/assets")
	for _, in := range []string{"", "/already/absolute.png", "https://cdn/x.png", "http://cdn/x.png"} {
		if got := s.publicAssetURL(in); got != in {
			t.Errorf("publicAssetURL(%q) = %q, want unchanged", in, got)
		}
	}
}

func TestPublicItemRewritesNestedAssets(t *testing.T) {
	s := assetServer("/assets")
	it := store.Item{ID: "i1", Assets: []store.Asset{
		{ID: "a1", URL: "42/ab12.png"},
		{ID: "a2", URL: ""},
	}}
	got := s.publicItem(it)
	if got.Assets[0].URL != "/assets/42/ab12.png" {
		t.Errorf("asset url = %q", got.Assets[0].URL)
	}
	if got.Assets[1].URL != "" {
		t.Errorf("an empty storage path must stay empty, got %q", got.Assets[1].URL)
	}
	// The input must not be mutated: the same slice is shared with the store's
	// caller in some paths.
	if it.Assets[0].URL != "42/ab12.png" {
		t.Error("publicItem mutated its argument")
	}
}

func TestPublicItemHandlesNilAssets(t *testing.T) {
	s := assetServer("/assets")
	got := s.publicItem(store.Item{ID: "i1"})
	if got.Assets == nil {
		t.Error("assets must serialise as [] not null, or the UI's .length breaks")
	}
	if len(got.Assets) != 0 {
		t.Errorf("got %d assets", len(got.Assets))
	}
}

func TestPublicAssetURLWithTrailingSlashBase(t *testing.T) {
	// config already TrimRight's the base, but a misconfigured value must not
	// produce a double slash.
	s := assetServer("/assets/")
	got := s.publicAssetURL("42/x.png")
	if strings.Contains(got, "//") {
		t.Errorf("double slash in %q", got)
	}
}
