package store

import (
	"strings"
	"testing"
)

func TestEscapeLike(t *testing.T) {
	cases := map[string]string{
		"":                "",
		"bottle":          "bottle",
		"100%":            `100\%`,
		"a_b":             `a\_b`,
		`back\slash`:      `back\\slash`,
		"%_%":             `\%\_\%`,
		"50%_off_now":     `50\%\_off\_now`,
		"already\\%done%": `already\\\%done\%`,
	}
	for in, want := range cases {
		if got := escapeLike(in); got != want {
			t.Errorf("escapeLike(%q) = %q, want %q", in, got, want)
		}
	}
}

// The list filter used `$1 <% sku || …`, and `<%` is not a PostgreSQL
// operator, so every GET /api/v1/items returned 500. These tests pin the
// replacement: a trigram-indexed generated column plus an escaped ILIKE.
func TestSearchUsesILikeNotTheNonexistentOperator(t *testing.T) {
	src := readStoreSource(t)
	if strings.Contains(src, "<%") {
		t.Error(`store.go still contains the invalid "<%" operator`)
	}
	if !strings.Contains(src, "search_text ILIKE") {
		t.Error("the list query must filter on the generated search_text column")
	}
	if !strings.Contains(src, `ESCAPE '\'`) {
		t.Error("the ILIKE must declare ESCAPE so escapeLike is meaningful")
	}
}

func readStoreSource(t *testing.T) string {
	t.Helper()
	body, err := readFile("store.go")
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestListItemsPageCursorRoundTrip(t *testing.T) {
	page := ListItemsPage{Seq: 4711}
	cursor := page.Cursor()
	if cursor != "4711" {
		t.Fatalf("Cursor() = %q", cursor)
	}
	if got := ParseListItemsPage(cursor); got != 4711 {
		t.Errorf("ParseListItemsPage = %d, want 4711", got)
	}
}

func TestListItemsPageZeroValueHasNoCursor(t *testing.T) {
	// A zero Seq means "no more pages", and must not render as "0", which
	// would make the client request a page that starts from the beginning.
	if c := (ListItemsPage{}).Cursor(); c != "" {
		t.Errorf("zero page cursor = %q, want empty", c)
	}
	if got := ParseListItemsPage("0"); got != 0 {
		t.Errorf(`ParseListItemsPage("0") = %d, want 0`, got)
	}
}

func TestParseListItemsPageTreatsGarbageAsStart(t *testing.T) {
	// A stale or hand-edited bookmark must not become a 500.
	for _, bad := range []string{"", "garbage", "abc|def", "-1", "1.5", "99999999999999999999"} {
		if got := ParseListItemsPage(bad); got != 0 {
			t.Errorf("ParseListItemsPage(%q) = %d; want 0", bad, got)
		}
	}
}

// The cursor must be a monotonic sequence, not (created_at, id): several items
// created in the same millisecond share a created_at, so a row comparison
// skipped the ones with a lower uuid.
func TestPaginationKeyIsMonotonic(t *testing.T) {
	src := readStoreSource(t)
	if strings.Contains(src, "created_at, id) >") {
		t.Error("pagination must not key off (created_at, id); it is not unique")
	}
	if !strings.Contains(src, "seq < $5") {
		t.Error("pagination must key off the monotonic items.seq column")
	}
	if !strings.Contains(src, "ORDER BY seq DESC") {
		t.Error("listing must be ordered by seq so the cursor is consistent")
	}
}
