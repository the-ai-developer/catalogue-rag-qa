import { For, Show, createSignal, onCleanup, onMount } from 'solid-js';
import { useSearchParams } from '@solidjs/router';
import {
  createDescriptionJob, getDescriptionJob, publishDescriptionJob, reviewDescriptionJob, toApiError,
} from '../api/client';
import type { DescriptionJob, SpecSheet } from '../api/types';
import { DraftCard, ErrorBanner, JobProgress, LoadingSkeleton, SpecForm } from '../components/ui';
import { POLL_INTERVAL_MS, pollUntil } from '../lib/poll';

const PENDING = new Set(['queued', 'running']);
const isPending = (status: string) => PENDING.has(status);

/**
 * Project 2 — Spec → Description with the editor gate.
 * Review actions and publish are only offered when the server state machine
 * allows them (draft_ready → review; approved → publish).
 */
export default function DescriptionsPage() {
  const [params, setParams] = useSearchParams<{ job?: string }>();
  const [job, setJob] = createSignal<DescriptionJob | null>(null);
  const [selectedDraft, setSelectedDraft] = createSignal<string | null>(null);
  const [editText, setEditText] = createSignal('');
  const [notes, setNotes] = createSignal('');
  const [showEdit, setShowEdit] = createSignal(false);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  let poller: { stop: () => void } | null = null;

  const watch = (id: string) => {
    poller?.stop();
    poller = pollUntil(
      () => getDescriptionJob(id),
      (j: DescriptionJob) => !isPending(j.status),
      POLL_INTERVAL_MS,
      {
        onValue: (j) => {
          setJob(j);
          const first = j.drafts[0];
          if (first && !selectedDraft()) setSelectedDraft(first.id);
        },
        onError: (e) => {
          setError(toApiError(e).message);
          return false;
        },
      },
    );
  };

  onMount(() => {
    const id = params.job;
    if (id) {
      setLoading(true);
      getDescriptionJob(id)
        .then((j) => {
          setJob(j);
          if (j.drafts[0]) setSelectedDraft(j.drafts[0].id);
          if (isPending(j.status)) watch(id);
        })
        .catch((e) => setError(toApiError(e).message))
        .finally(() => setLoading(false));
    }
  });
  onCleanup(() => poller?.stop());

  const generate = async (spec: SpecSheet, beamWidth: number, maxLen: number) => {
    setLoading(true);
    setError(null);
    try {
      const enq = await createDescriptionJob({ spec, beam_width: beamWidth, max_len: maxLen });
      setParams({ job: enq.job_id });
      setJob(null);
      watch(enq.job_id);
    } catch (e) {
      setError(toApiError(e).message);
      setLoading(false);
    }
  };

  const act = async (decision: 'approve' | 'reject' | 'edit') => {
    const j = job();
    if (!j) return;
    setLoading(true);
    setError(null);
    try {
      const req = { decision, draft_id: selectedDraft() ?? undefined, edited_text: editText() || undefined, notes: notes() || undefined };
      setJob(await reviewDescriptionJob(j.id, req));
      setShowEdit(false);
    } catch (e) {
      setError(toApiError(e).message);
    } finally {
      setLoading(false);
    }
  };

  const publish = async () => {
    const j = job();
    if (!j) return;
    setLoading(true);
    try {
      await publishDescriptionJob(j.id);
      setJob(await getDescriptionJob(j.id));
    } catch (e) {
      setError(toApiError(e).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section>
      <h2 class="page-title">Spec → Description</h2>
      <p class="muted">Beam-search drafts from the fine-tuned encoder-decoder. Nothing reaches the client until an editor approves it.</p>
      <ErrorBanner message={error()} onDismiss={() => setError(null)} />

      <SpecForm pending={loading()} onSubmit={generate} />

      <Show when={!loading() || job()} fallback={<LoadingSkeleton label="Generating drafts…" />}>
        <Show when={job()}>
          {(j) => (
            <div class="card">
              <header>
                <h3>Job #{j().id.slice(0, 8)}</h3>
                <JobProgress status={j().status} />
              </header>
              <p class="muted small">
                beam {j().beam_width} · max {j().max_len}
                <Show when={j().model_version}> · {j().model_name}@{j().model_version}</Show>
              </p>

              <Show when={j().status === 'draft_ready'}>
                <h4>Ranked drafts — pick one</h4>
                <For each={j().drafts}>
                  {(d) => (
                    <DraftCard draft={d} selected={selectedDraft() === d.id}
                      onSelect={setSelectedDraft} />
                  )}
                </For>
                <div class="row">
                  <button class="btn btn-primary" disabled={loading() || !selectedDraft()} onClick={() => act('approve')}>
                    Approve draft
                  </button>
                  <button class="btn" disabled={loading() || !selectedDraft()} onClick={() => setShowEdit(!showEdit())}>
                    Edit inline
                  </button>
                  <button class="btn btn-danger" disabled={loading()} onClick={() => act('reject')}>
                    Reject
                  </button>
                </div>
                <Show when={showEdit()}>
                  <form class="form" onSubmit={(e) => { e.preventDefault(); act('edit'); }}>
                    <label>Edited text
                      <textarea class="input" rows={4} value={editText()}
                        onInput={(e) => setEditText(e.currentTarget.value)}
                        placeholder="Rewrite the draft before approving…" />
                    </label>
                    <button class="btn btn-primary" disabled={loading() || !editText().trim()}>
                      Save edit & approve
                    </button>
                  </form>
                </Show>
                <label>Review notes
                  <input class="input" value={notes()} onInput={(e) => setNotes(e.currentTarget.value)} placeholder="optional" />
                </label>
              </Show>

              <Show when={j().status === 'approved'}>
                <div class="banner banner-warn">
                  <strong>Approved — not yet published.</strong>
                </div>
                <button class="btn btn-primary" disabled={loading()} onClick={publish}>
                  Publish to client
                </button>
              </Show>

              <Show when={j().status === 'published'}>
                <div class="banner banner-ok">Published ✅</div>
              </Show>

              <Show when={j().status === 'rejected'}>
                <div class="banner banner-error">Rejected by editor.</div>
              </Show>

              <Show when={j().error}>
                <div class="banner banner-error">{j().error}</div>
              </Show>

              <h4>Reviews</h4>
              <For each={j().reviews} fallback={<p class="muted small">No reviews yet.</p>}>
                {(r) => (
                  <p class="small">
                    <strong>{r.decision}</strong> by {r.reviewer}
                    <Show when={r.edited_text}> · edited: “{r.edited_text!.slice(0, 60)}…”</Show>
                    <Show when={r.notes}> · {r.notes}</Show>
                  </p>
                )}
              </For>
            </div>
          )}
        </Show>
      </Show>
    </section>
  );
}
