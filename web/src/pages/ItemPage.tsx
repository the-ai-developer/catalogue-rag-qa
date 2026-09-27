import { For, Show, createSignal, onCleanup, onMount } from 'solid-js';
import { useParams } from '@solidjs/router';
import {
  enqueueIngest, getIngestJob, getItem, getPublishedDescriptions, toApiError, uploadAsset,
} from '../api/client';
import type { IngestJob, Item, PublishedDescription } from '../api/types';
import {
  ErrorBanner, JobProgress, LoadingSkeleton, StatusBadge, UploadDropzone,
} from '../components/ui';
import { formatJSON } from '../lib/format';
import { INGEST_TERMINAL, POLL_INTERVAL_MS, pollUntil } from '../lib/poll';

export default function ItemPage() {
  const params = useParams<{ id: string }>();
  const [item, setItem] = createSignal<Item | null>(null);
  const [job, setJob] = createSignal<IngestJob | null>(null);
  const [published, setPublished] = createSignal<PublishedDescription[]>([]);
  const [error, setError] = createSignal<string | null>(null);
  const [uploading, setUploading] = createSignal(false);
  let poller: { stop: () => void } | null = null;

  const refresh = async () => {
    try {
      setItem(await getItem(params.id));
      setPublished((await getPublishedDescriptions(params.id)).items);
      setError(null);
    } catch (e) {
      setError(toApiError(e).message);
    }
  };

  onMount(refresh);
  onCleanup(() => poller?.stop());

  const startIngest = async () => {
    try {
      const enq = await enqueueIngest(params.id);
      setJob({ id: enq.job_id, item_id: params.id, status: 'queued', chunks_indexed: 0, images_indexed: 0, error: null } as IngestJob);
      poller?.stop();
      poller = pollUntil(
        () => getIngestJob(enq.job_id),
        (j: IngestJob) => INGEST_TERMINAL.has(j.status),
        POLL_INTERVAL_MS,
        {
          onValue: (j) => {
            setJob(j);
            // Counts and assets changed, so refresh the item once it settles.
            if (INGEST_TERMINAL.has(j.status)) void refresh();
          },
          onError: (e) => {
            setError(toApiError(e).message);
            return false; // stop: the job may not exist
          },
        },
      );
    } catch (e) {
      setError(toApiError(e).message);
    }
  };

  const upload = async (file: File) => {
    setUploading(true);
    try {
      await uploadAsset(params.id, file);
      await refresh();
    } catch (e) {
      setError(toApiError(e).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <section>
      {/* Outside the Show on purpose: nesting the banner inside meant a failed
          load rendered neither the item nor the error, just an eternal
          "Loading item…" skeleton. */}
      <ErrorBanner message={error()} onDismiss={() => setError(null)} />
      <Show when={item()} fallback={<LoadingSkeleton label="Loading item…" />}>
        {(it) => (
          <>
            <h2 class="page-title">{it().title} <StatusBadge status={it().status} /></h2>
            <div class="card">
              <table class="quotes">
                <tbody>
                  <tr><th>SKU</th><td>{it().sku}</td></tr>
                  <tr><th>Category</th><td>{it().category}</td></tr>
                  <tr><th>Material</th><td>{it().material ?? '—'}</td></tr>
                  <tr><th>Dimensions</th><td>{formatJSON(it().dimensions)}</td></tr>
                  <tr><th>Features</th><td>{(it().features ?? []).join(', ') || '—'}</td></tr>
                  <tr><th>Indexed</th><td>{it().counts.chunks} chunks · {it().counts.embeddings_text} text · {it().counts.embeddings_image} image</td></tr>
                </tbody>
              </table>
            </div>

            <h3>Product photos</h3>
            <UploadDropzone onFile={upload} pending={uploading()} />
            <div class="grid">
              <For each={it().assets} fallback={<p class="muted">No assets yet.</p>}>
                {(a) => (
                  <figure class="card">
                    <img src={a.url} alt={a.mime} width="160" loading="lazy" />
                    <figcaption class="small muted">{a.mime} · {a.bytes} bytes</figcaption>
                  </figure>
                )}
              </For>
            </div>

            <h3>Ingest (chunk → dual embeddings → FAISS)</h3>
            <Show when={job()} fallback={<button class="btn btn-primary" onClick={startIngest}>Run ingest</button>}>
              {(j) => <JobProgress status={j().status} />}
            </Show>

            <h3>Published descriptions</h3>
            <For each={published()} fallback={<p class="muted">None yet — generate and approve one on the Descriptions page.</p>}>
              {(p) => <div class="card"><p>{p.text}</p><p class="muted small">approved by {p.approved_by}</p></div>}
            </For>
          </>
        )}
      </Show>
    </section>
  );
}
