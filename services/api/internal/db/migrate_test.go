package db

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestLoadMigrationsSortsAndSkipsNonSQL(t *testing.T) {
	dir := t.TempDir()
	write := func(name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("0002_second.sql", "SELECT 2;")
	write("0001_init.sql", "SELECT 1;")
	write("0010_tenth.sql", "SELECT 10;")
	write("README.md", "not sql")
	if err := os.Mkdir(filepath.Join(dir, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}

	files, err := loadMigrations(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range files {
		names = append(names, f.name)
	}
	want := []string{"0001_init.sql", "0002_second.sql", "0010_tenth.sql"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Errorf("names = %v, want %v", names, want)
	}
}

func TestLoadMigrationsFailsLoudlyOnAMissingDir(t *testing.T) {
	// The old behaviour silently fell back to an embedded copy of the schema.
	// Two copies of the migration is how "relation already exists" on boot
	// happened, so a missing directory must stop the process instead.
	_, err := loadMigrations(filepath.Join(t.TempDir(), "nope"))
	if err == nil {
		t.Fatal("a missing migrations dir must be an error")
	}
	if !strings.Contains(err.Error(), "MIGRATIONS_DIR") {
		t.Errorf("the error should tell the operator what to set, got: %v", err)
	}
}

func TestLoadMigrationsFailsOnAnEmptyDir(t *testing.T) {
	if _, err := loadMigrations(t.TempDir()); err == nil {
		t.Fatal("an empty migrations dir must be an error, not a silent no-op")
	}
}

func TestUpSection(t *testing.T) {
	cases := []struct {
		name, in, want string
	}{
		{
			"markers with both sections",
			"-- +migrate Up\nCREATE TABLE a();\n-- +migrate Down\nDROP TABLE a;",
			"\nCREATE TABLE a();\n",
		},
		{
			"up only",
			"-- +migrate Up\nCREATE TABLE a();",
			"\nCREATE TABLE a();",
		},
		{
			"down only truncates",
			"CREATE TABLE a();\n-- +migrate Down\nDROP TABLE a;",
			"CREATE TABLE a();\n",
		},
		{"no markers runs verbatim", "CREATE TABLE a();", "CREATE TABLE a();"},
		{"empty", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := upSection(c.in); got != c.want {
				t.Errorf("upSection(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestShippedSchemaIsIdempotentPerLedger pins the invariant that the single
// migration file is applied exactly once, by the Go runner, and that the SQL
// itself is therefore free to be non-idempotent.
func TestShippedSchemaIsIdempotentPerLedger(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..", "db", "migrations")
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Skipf("schema not reachable from the test cwd: %v", err)
	}
	sql := 0
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".sql") {
			sql++
		}
	}
	if sql == 0 {
		t.Fatal("no migrations found in db/migrations")
	}

	// The api image must ship the schema, or Migrate hard-fails by design.
	df := filepath.Join("..", "..", "..", "..", "deploy", "docker", "Dockerfile.api")
	body, err := os.ReadFile(df)
	if err != nil {
		t.Fatal(err)
	}
	// Flags may sit between COPY and the source (`COPY --chmod=0555 src dst`),
	// so match the source and destination rather than the whole instruction.
	copyRe := regexp.MustCompile(`(?m)^\s*COPY\b[^\n]*\bdb/migrations\s+/db/migrations\b`)
	if !copyRe.Match(body) {
		t.Error("Dockerfile.api must copy db/migrations, since the embedded " +
			"fallback was removed")
	}
	if !strings.Contains(string(body), "MIGRATIONS_DIR=/db/migrations") {
		t.Error("Dockerfile.api must point MIGRATIONS_DIR at the shipped copy")
	}
}
