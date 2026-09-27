// Package apperr defines the canonical API error envelope shared by all
// packages of the Go API service.
//
// Envelope (api-contract.md §Conventions):
//
//	{ "error": { "code": "not_found", "message": "item not found", "details": {} } }
//
// Codes: bad_request, unauthorized, forbidden, not_found, conflict,
// unprocessable, upstream, internal.
package apperr

import (
	"errors"
	"fmt"
)

// Code is an API error code from the contract.
type Code string

const (
	BadRequest    Code = "bad_request"
	Unauthorized  Code = "unauthorized"
	Forbidden     Code = "forbidden"
	NotFound      Code = "not_found"
	Conflict      Code = "conflict"
	Unprocessable Code = "unprocessable"
	Upstream      Code = "upstream"
	Internal      Code = "internal"
)

// Error is a structured API error. It is returned by services and rendered by
// the HTTP layer into the contract error envelope.
type Error struct {
	Code    Code
	Message string
	Details map[string]any
	Err     error
}

// ErrAuthUnavailable marks a failure to *consult* the key store (database down,
// pool exhausted, context deadline). It is distinct from "no such key" so a
// database blip returns 503 instead of a 401 that looks like a bad credential
// and silently signs every user out.
var ErrAuthUnavailable = errors.New("key store unavailable")

func (e *Error) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %s: %v", e.Code, e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *Error) Unwrap() error { return e.Err }

// New builds an Error without an underlying cause.
func New(code Code, message string) *Error {
	return &Error{Code: code, Message: message, Details: map[string]any{}}
}

// Wrap builds an Error with an underlying cause (kept with %w semantics).
func Wrap(code Code, message string, err error) *Error {
	return &Error{Code: code, Message: message, Details: map[string]any{}, Err: err}
}

// WithDetail attaches a detail entry (fluent).
func (e *Error) WithDetail(key string, value any) *Error {
	if e.Details == nil {
		e.Details = map[string]any{}
	}
	e.Details[key] = value
	return e
}

// Constructors for the frequent cases.
func BadRequestErr(msg string) *Error    { return New(BadRequest, msg) }
func UnauthorizedErr(msg string) *Error  { return New(Unauthorized, msg) }
func ForbiddenErr(msg string) *Error     { return New(Forbidden, msg) }
func NotFoundErr(msg string) *Error      { return New(NotFound, msg) }
func ConflictErr(msg string) *Error      { return New(Conflict, msg) }
func UnprocessableErr(msg string) *Error { return New(Unprocessable, msg) }
func InternalErr(msg string, err error) *Error {
	return Wrap(Internal, msg, err)
}

// AsError extracts an *Error from an error chain, if present.
func AsError(err error) (*Error, bool) {
	var e *Error
	if errors.As(err, &e) {
		return e, true
	}
	return nil, false
}
