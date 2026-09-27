package workflow

import (
	"errors"
	"testing"
)

func strp(s string) *string { return &s }

func TestCanReviewOnlyFromDraftReady(t *testing.T) {
	for _, status := range []string{JobDraftReady} {
		for _, d := range []string{DecisionApprove, DecisionReject, DecisionEdit} {
			if !CanReview(status, d) {
				t.Errorf("CanReview(%s, %s) = false, want true", status, d)
			}
		}
	}
	for _, status := range []string{JobQueued, JobRunning, JobApproved, JobRejected,
		JobPublished, JobFailed, "unknown"} {
		for _, d := range []string{DecisionApprove, DecisionReject, DecisionEdit} {
			if CanReview(status, d) {
				t.Errorf("CanReview(%s, %s) = true, want false", status, d)
			}
		}
	}
}

func TestCanReviewRejectsUnknownDecisions(t *testing.T) {
	for _, d := range []string{"", "publish", "APPROVE", "delete", "approve "} {
		if CanReview(JobDraftReady, d) {
			t.Errorf("CanReview(draft_ready, %q) = true, want false", d)
		}
	}
}

func TestReviewOutcome(t *testing.T) {
	cases := map[string]string{
		DecisionApprove: JobApproved,
		DecisionEdit:    JobApproved,
		DecisionReject:  JobRejected,
	}
	for d, want := range cases {
		got, ok := ReviewOutcome(d)
		if !ok || got != want {
			t.Errorf("ReviewOutcome(%q) = %q,%v want %q", d, got, ok, want)
		}
	}
	if _, ok := ReviewOutcome("publish"); ok {
		t.Error("publish is not a review decision")
	}
}

func TestCanPublishOnlyFromApproved(t *testing.T) {
	if !CanPublish(JobApproved) {
		t.Error("approved jobs must be publishable")
	}
	for _, status := range []string{JobQueued, JobRunning, JobDraftReady,
		JobRejected, JobPublished, JobFailed} {
		if CanPublish(status) {
			t.Errorf("CanPublish(%s) = true, want false", status)
		}
	}
}

// --- the editor gate ------------------------------------------------------- //
// ApprovedText is the single place that decides what text a human signed off
// on. The bug it replaces fell back to Drafts[0] — the raw model output — when
// no review matched, which published unreviewed text.

func TestApprovedTextPrefersTheLatestEdit(t *testing.T) {
	drafts := []Draft{{ID: "d1", Text: "model draft one"},
		{ID: "d2", Text: "model draft two"}}
	reviews := []Review{
		{Decision: DecisionApprove, DraftID: strp("d1")},
		{Decision: DecisionEdit, DraftID: strp("d2"), EditedText: strp("human rewrite")},
	}
	text, draftID, err := ApprovedText(reviews, drafts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if text != "human rewrite" {
		t.Errorf("text = %q, want the editor's rewrite", text)
	}
	if draftID != "d2" {
		t.Errorf("draftID = %q, want d2", draftID)
	}
}

func TestApprovedTextFallsBackToTheApprovedDraft(t *testing.T) {
	drafts := []Draft{{ID: "d1", Text: "model draft one"}}
	reviews := []Review{{Decision: DecisionApprove, DraftID: strp("d1")}}
	text, draftID, err := ApprovedText(reviews, drafts)
	if err != nil || text != "model draft one" || draftID != "d1" {
		t.Fatalf("text=%q draft=%q err=%v", text, draftID, err)
	}
}

func TestApprovedTextNeverFallsBackToAnUnreviewedDraft(t *testing.T) {
	// The exact bypass: drafts exist, the job claims approved, but no review
	// approved anything. Publishing must fail, not fall back to draft[0].
	drafts := []Draft{{ID: "d1", Text: "hallucinated model output"}}
	cases := map[string][]Review{
		"no reviews":    {},
		"only a reject": {{Decision: DecisionReject}},
		"approve no id": {{Decision: DecisionApprove}},
		"unknown draft": {{Decision: DecisionApprove, DraftID: strp("nope")}},
		"empty edit":    {{Decision: DecisionEdit, DraftID: strp("d1"), EditedText: strp("   ")}},
		"nil edit":      {{Decision: DecisionEdit, DraftID: strp("d1")}},
	}
	for name, reviews := range cases {
		t.Run(name, func(t *testing.T) {
			if _, _, err := ApprovedText(reviews, drafts); !errors.Is(err, ErrNoApprovedText) {
				t.Fatalf("err = %v, want ErrNoApprovedText", err)
			}
		})
	}
}

func TestApprovedTextWithNoDraftsAtAll(t *testing.T) {
	if _, _, err := ApprovedText(nil, nil); !errors.Is(err, ErrNoApprovedText) {
		t.Fatalf("err = %v, want ErrNoApprovedText", err)
	}
}

func TestApprovedTextSkipsAnEmptyEditAndUsesAnEarlierApproval(t *testing.T) {
	drafts := []Draft{{ID: "d1", Text: "approved text"}}
	reviews := []Review{
		{Decision: DecisionApprove, DraftID: strp("d1")},
		{Decision: DecisionEdit, DraftID: strp("d1"), EditedText: strp("")},
	}
	text, _, err := ApprovedText(reviews, drafts)
	if err != nil || text != "approved text" {
		t.Fatalf("text=%q err=%v", text, err)
	}
}

// --- transition guard ------------------------------------------------------ //

func TestCanTransition(t *testing.T) {
	legal := [][2]string{
		{JobQueued, JobRunning}, {JobQueued, JobFailed},
		{JobRunning, JobDraftReady}, {JobRunning, JobFailed},
		{JobDraftReady, JobApproved}, {JobDraftReady, JobRejected},
		{JobApproved, JobPublished},
	}
	for _, p := range legal {
		if !CanTransition(p[0], p[1]) {
			t.Errorf("CanTransition(%s, %s) = false, want true", p[0], p[1])
		}
	}
	// Terminal states are terminal, and draft_ready cannot jump to published.
	illegal := [][2]string{
		{JobQueued, JobDraftReady}, {JobQueued, JobPublished},
		{JobDraftReady, JobPublished}, {JobRunning, JobApproved},
		{JobRejected, JobApproved}, {JobPublished, JobApproved},
		{JobFailed, JobQueued}, {JobFailed, JobRunning},
		{JobApproved, JobDraftReady},
	}
	for _, p := range illegal {
		if CanTransition(p[0], p[1]) {
			t.Errorf("CanTransition(%s, %s) = true, want false", p[0], p[1])
		}
	}
}

func TestAssertTransitionMessage(t *testing.T) {
	if err := AssertTransition(JobDraftReady, JobApproved); err != nil {
		t.Fatalf("legal transition rejected: %v", err)
	}
	err := AssertTransition(JobDraftReady, JobPublished)
	if err == nil {
		t.Fatal("expected an error")
	}
	if got := err.Error(); got != "illegal job transition draft_ready -> published" {
		t.Errorf("message = %q", got)
	}
}

func TestStatusConstantsMatchTheDatabaseCheck(t *testing.T) {
	// These strings are constrained by CHECK constraints in
	// db/migrations/0001_init.sql; drift here is a runtime insert failure.
	desc := map[string]bool{JobQueued: true, JobRunning: true, JobDraftReady: true,
		JobApproved: true, JobRejected: true, JobPublished: true, JobFailed: true}
	ingest := map[string]bool{IngestQueued: true, IngestRunning: true,
		IngestSucceeded: true, IngestFailed: true}
	if len(desc) != 7 || len(ingest) != 4 {
		t.Fatalf("duplicate status constants: %d desc, %d ingest", len(desc), len(ingest))
	}
}
