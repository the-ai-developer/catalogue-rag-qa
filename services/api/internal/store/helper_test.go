package store

import "os"

// readFile is a tiny test helper so the source-level assertions do not each
// import os.
func readFile(name string) (string, error) {
	b, err := os.ReadFile(name)
	return string(b), err
}
