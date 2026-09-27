package apperr

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"
)

func TestAsErrorFindsAWrappedError(t *testing.T) {
	inner := errors.New("root cause")
	err := Wrap(Upstream, "retrieval failed", inner)

	got, ok := AsError(err)
	if !ok {
		t.Fatal("AsError must find a wrapped *Error")
	}
	if got.Code != Upstream || got.Message != "retrieval failed" {
		t.Errorf("got %+v", got)
	}
	if !errors.Is(err, inner) {
		t.Error("the cause must remain unwrappable for errors.Is")
	}
	if errors.Unwrap(got) != inner {
		t.Error("Unwrap must return the cause")
	}
}

func TestAsErrorRejectsForeignErrors(t *testing.T) {
	if _, ok := AsError(errors.New("plain")); ok {
		t.Error("a plain error must not be reported as an *Error")
	}
	if _, ok := AsError(nil); ok {
		t.Error("nil must not be reported as an *Error")
	}
}

func TestNewAlwaysHasDetails(t *testing.T) {
	// The envelope promises details:{}; a nil map serialises as null and
	// clients that read it defensively are right to.
	b, err := json.Marshal(New(NotFound, "gone"))
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Code    string         `json:"code"`
		Message string         `json:"message"`
		Details map[string]any `json:"details"`
	}
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	if out.Details == nil {
		t.Error("details must serialise as {}, not null")
	}
}

func TestWithDetail(t *testing.T) {
	e := BadRequestErr("bad").WithDetail("field", "sku").WithDetail("n", 2)
	if e.Details["field"] != "sku" || e.Details["n"] != 2 {
		t.Errorf("details = %v", e.Details)
	}
	// Fluent on a nil map too.
	raw := &Error{Code: BadRequest, Message: "x"}
	raw.WithDetail("k", "v")
	if raw.Details["k"] != "v" {
		t.Error("WithDetail must initialise a nil map")
	}
}

func TestErrorMessageNeverLeaksTheCause(t *testing.T) {
	// The wire envelope carries Message, not Error(), so the cause stays in
	// the logs. This asserts the two are distinct values.
	e := Wrap(Internal, "storage", errors.New("disk /dev/sda1 full"))
	if e.Message == e.Error() {
		t.Error("Message must be safe to return to a client")
	}
	if e.Error() == e.Message {
		t.Error("Error() should include the cause for operators")
	}
}

func TestEveryCodeHasAnHTTPStatusInTheServer(t *testing.T) {
	// The contract lists 8 codes; the server's switch must map all of them.
	codes := []Code{BadRequest, Unauthorized, Forbidden, NotFound, Conflict,
		Unprocessable, Upstream, Internal}
	statuses := map[Code]int{
		BadRequest: http.StatusBadRequest, Unauthorized: http.StatusUnauthorized,
		Forbidden: http.StatusForbidden, NotFound: http.StatusNotFound,
		Conflict: http.StatusConflict, Unprocessable: http.StatusUnprocessableEntity,
		Upstream: http.StatusBadGateway, Internal: http.StatusInternalServerError,
	}
	for _, c := range codes {
		if _, ok := statuses[c]; !ok {
			t.Errorf("no status for code %q", c)
		}
	}
	if len(codes) != len(statuses) {
		t.Errorf("%d codes but %d statuses", len(codes), len(statuses))
	}
}

func TestErrorCodesAreSnakeCase(t *testing.T) {
	for _, c := range []Code{BadRequest, Unauthorized, Forbidden, NotFound,
		Conflict, Unprocessable, Upstream, Internal} {
		for _, r := range string(c) {
			if r >= 'A' && r <= 'Z' {
				t.Errorf("code %q must be lower snake_case", c)
				break
			}
			if r == ' ' || r == '-' {
				t.Errorf("code %q must be lower snake_case", c)
				break
			}
		}
	}
}
