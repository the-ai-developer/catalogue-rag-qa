// Package workflow holds the pure state machines of the product rules:
// the description editor gate and the ingest/generation job lifecycles.
//
// Nothing here touches the database, so every rule below is directly testable
// and can be reused as a guard inside a transaction.
package workflow

import (
	"errors"
	"fmt"
	"strings"
)

// Description job statuses (db: generation_jobs.status).
const (
	JobQueued     = "queued"
	JobRunning    = "running"
	JobDraftReady = "draft_ready"
	JobApproved   = "approved"
	JobRejected   = "rejected"
	JobPublished  = "published"
	JobFailed     = "failed"
)

// Review decisions (db: editor_reviews.decision).
const (
	DecisionApprove = "approve"
	DecisionReject  = "reject"
	DecisionEdit    = "edit"
)

// Ingest job statuses.
const (
	IngestQueued    = "queued"
	IngestRunning   = "running"
	IngestSucceeded = "succeeded"
	IngestFailed    = "failed"
)

// ErrNoApprovedText is returned when a job claims to be approved but carries no
// human-approved text. Publishing the raw model draft here would be exactly the
// gate bypass the product rules forbid.
var ErrNoApprovedText = errors.New("no human-approved text for this job")

// CanReview reports whether a human decision is allowed in the current status.
// Editor gate: only a finished draft can be reviewed.
func CanReview(status, decision string) bool {
	if status != JobDraftReady {
		return false
	}
	switch decision {
	case DecisionApprove, DecisionReject, DecisionEdit:
		return true
	default:
		return false
	}
}

// ReviewOutcome maps a decision to the next status.
func ReviewOutcome(decision string) (string, bool) {
	switch decision {
	case DecisionApprove, DecisionEdit:
		return JobApproved, true
	case DecisionReject:
		return JobRejected, true
	default:
		return "", false
	}
}

// CanPublish enforces that only approved descriptions reach the client.
func CanPublish(status string) bool { return status == JobApproved }

// Draft is the minimum a Draft lookup needs.
type Draft struct {
	ID   string
	Text string
}

// Review is the minimum a Review lookup needs, newest last.
type Review struct {
	Decision   string
	DraftID    *string
	EditedText *string
}

// ApprovedText returns the text a human signed off on: the most recent edit if
// there is one, otherwise the approved draft. It never falls back to an
// unreviewed draft — that is the editor gate.
//
// Returns ErrNoApprovedText when no review approved anything, so the caller
// refuses to publish rather than shipping model output as if a human wrote it.
func ApprovedText(reviews []Review, drafts []Draft) (text string, draftID string, err error) {
	for i := len(reviews) - 1; i >= 0; i-- {
		rv := reviews[i]
		switch rv.Decision {
		case DecisionEdit:
			if rv.EditedText != nil && strings.TrimSpace(*rv.EditedText) != "" {
				id := ""
				if rv.DraftID != nil {
					id = *rv.DraftID
				}
				return strings.TrimSpace(*rv.EditedText), id, nil
			}
		case DecisionApprove:
			if rv.DraftID == nil {
				continue
			}
			for _, d := range drafts {
				if d.ID == *rv.DraftID {
					return d.Text, d.ID, nil
				}
			}
		}
	}
	return "", "", ErrNoApprovedText
}

// CanTransition is the generic job progress guard for workers and handlers.
func CanTransition(from, to string) bool {
	allowed := map[string][]string{
		JobQueued:     {JobRunning, JobFailed},
		JobRunning:    {JobDraftReady, JobFailed},
		JobDraftReady: {JobApproved, JobRejected},
		JobApproved:   {JobPublished},
		JobRejected:   {},
		JobPublished:  {},
		JobFailed:     {},
	}
	for _, next := range allowed[from] {
		if next == to {
			return true
		}
	}
	return false
}

// AssertTransition returns a descriptive error when a transition is illegal.
// Used inside the same transaction as the write it guards.
func AssertTransition(from, to string) error {
	if CanTransition(from, to) {
		return nil
	}
	return fmt.Errorf("illegal job transition %s -> %s", from, to)
}

// Ingest job transitions are guarded in SQL instead: FinishIngestJob and
// FinishGenerationJob both write `WHERE status='running'`, so the database
// rejects an illegal move atomically. Duplicating that check in Go would give
// two places to keep in sync and neither would be the one that actually holds.
