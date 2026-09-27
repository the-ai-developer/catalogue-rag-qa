/**
 * Shared UI components. Product rules live here too:
 * CitationList refuses to render ungrounded answers (Project 1) and the
 * DescriptionsPage only offers review/publish when the server allows it (P2).
 */
import { For, Show, createSignal } from 'solid-js';
import type { Citation, Draft, Item, SpecSheet } from '../api/types';

const TONE: Record<string, string> = {
  draft: 'neutral', queued: 'neutral', running: 'info',
  active: 'ok', succeeded: 'ok', draft_ready: 'info', approved: 'ok',
  published: 'ok', rejected: 'bad', failed: 'bad', archived: 'neutral',
};

export function StatusBadge(props: { status: string }) {
  return <span class={`badge badge-${TONE[props.status] ?? 'neutral'}`}>{props.status.replace(/_/g, ' ')}</span>;
}

export function ErrorBanner(props: { message: string | null; onDismiss?: () => void }) {
  return (
    <Show when={props.message}>
      <div class="banner banner-error" role="alert">
        <span>{props.message}</span>
        <Show when={props.onDismiss}>
          <button class="banner-close" onClick={() => props.onDismiss?.()} aria-label="dismiss">×</button>
        </Show>
      </div>
    </Show>
  );
}

export function LoadingSkeleton(props: { label?: string }) {
  return <div class="skeleton" aria-busy="true">{props.label ?? 'Loading…'}</div>;
}

export function SearchBar(props: { value: string; onInput: (v: string) => void; placeholder?: string }) {
  return (
    <input
      class="input search" type="search" value={props.value}
      placeholder={props.placeholder ?? 'Search catalogue…'}
      onInput={(e) => props.onInput(e.currentTarget.value)}
    />
  );
}

export function ItemCard(props: { item: Item; onOpen: (id: string) => void }) {
  return (
    <article class="card item-card" onClick={() => props.onOpen(props.item.id)}
      onKeyDown={(e) => e.key === 'Enter' && props.onOpen(props.item.id)}
      tabindex="0" role="button">
      <header>
        <strong>{props.item.title}</strong>
        <StatusBadge status={props.item.status} />
      </header>
      <p class="muted">{props.item.sku} · {props.item.category}
        <Show when={props.item.material}> · {props.item.material}</Show>
      </p>
      <footer class="muted small">
        {props.item.counts.chunks} chunks · {props.item.counts.embeddings_text} text ·
        {' '}{props.item.counts.embeddings_image} image
      </footer>
    </article>
  );
}

/**
 * Citation sidebar. If the citation check failed we MUST NOT present the
 * answer as grounded — we show an explicit withheld state instead.
 */
export function CitationList(props: { citations: Citation[]; passed: boolean }) {
  return (
    <section class="citations" aria-label="citations">
      <Show
        when={props.passed}
        fallback={
          <div class="banner banner-warn" role="alert">
            <strong>Ungrounded answer withheld.</strong>
            <p>The citation check failed — retrieved evidence does not support every sentence,
              so the answer is not displayed.</p>
          </div>
        }>
        <h3>Evidence</h3>
        <Show when={props.citations.length} fallback={<p class="muted">No citations returned.</p>}>
          <ul>
            <For each={props.citations}>
              {(c) => (
                <li class="citation">
                  <header>
                    <span class={`badge badge-${c.modality === 'image' ? 'info' : 'neutral'}`}>{c.modality}</span>
                    <code>{c.sku}</code>
                    <span class="muted small">#{c.sentence_index} · score {c.score.toFixed(2)}</span>
                  </header>
                  <p class="snippet">{c.snippet}</p>
                </li>
              )}
            </For>
          </ul>
        </Show>
      </Show>
    </section>
  );
}

export function DraftCard(props: {
  draft: Draft; selected: boolean; onSelect: (id: string) => void;
}) {
  return (
    <label class={`card draft-card ${props.selected ? 'selected' : ''}`}>
      <header>
        <input type="radio" name="draft" checked={props.selected}
          onChange={() => props.onSelect(props.draft.id)} />
        <strong>Rank {props.draft.rank}</strong>
        <span class="muted small">score {props.draft.score.toFixed(2)}</span>
      </header>
      <p>{props.draft.text}</p>
    </label>
  );
}

export function JobProgress(props: { status: string }) {
  const pending = () => props.status === 'queued' || props.status === 'running';
  return (
    <div class="job-progress">
      <StatusBadge status={props.status} />
      <Show when={pending()}><span class="pulse">working…</span></Show>
    </div>
  );
}

export function UploadDropzone(props: { onFile: (f: File) => void; pending: boolean }) {
  return (
    <label class="dropzone">
      <input type="file" accept="image/png,image/jpeg" disabled={props.pending}
        onChange={(e) => {
          const f = e.currentTarget.files?.[0];
          if (f) props.onFile(f);
          e.currentTarget.value = '';
        }} />
      <Show when={props.pending} fallback={<span>Drop a product photo or click to upload (png/jpg)</span>}>
        <span>Uploading…</span>
      </Show>
    </label>
  );
}

/** Spec sheet form (Project 2 input: category/material/dimensions/features). */
export function SpecForm(props: {
  pending: boolean;
  onSubmit: (spec: SpecSheet, beamWidth: number, maxLen: number) => void;
}) {
  const [title, setTitle] = createSignal('');
  const [category, setCategory] = createSignal('');
  const [material, setMaterial] = createSignal('');
  const [dims, setDims] = createSignal('');
  const [features, setFeatures] = createSignal('');
  const [beam, setBeam] = createSignal(4);
  const [maxLen, setMaxLen] = createSignal(192);
  const [error, setError] = createSignal<string | null>(null);

  const parseDims = (raw: string): Record<string, number> => {
    const out: Record<string, number> = {};
    for (const pair of raw.split(',')) {
      const [k, v] = pair.split('=').map((s) => s.trim());
      if (k && v && !Number.isNaN(Number(v))) out[k] = Number(v);
    }
    return out;
  };

  const submit = () => {
    if (!category().trim()) {
      setError('category is required');
      return;
    }
    setError(null);
    const spec: SpecSheet = { category: category().trim() };
    if (title().trim()) spec.title = title().trim();
    if (material().trim()) spec.material = material().trim();
    const d = parseDims(dims());
    if (Object.keys(d).length) spec.dimensions = d;
    const f = features().split(',').map((s) => s.trim()).filter(Boolean);
    if (f.length) spec.features = f;
    props.onSubmit(spec, beam(), maxLen());
  };

  return (
    <form class="card form" onSubmit={(e) => { e.preventDefault(); submit(); }}>
      <Show when={error()}><div class="banner banner-error">{error()}</div></Show>
      <label>Title<input class="input" value={title()} onInput={(e) => setTitle(e.currentTarget.value)} placeholder="Kestrel 12 oz Insulated Bottle" /></label>
      <label>Category *<input class="input" value={category()} onInput={(e) => setCategory(e.currentTarget.value)} placeholder="drinkware" required /></label>
      <label>Material<input class="input" value={material()} onInput={(e) => setMaterial(e.currentTarget.value)} placeholder="18/8 stainless steel" /></label>
      <label>Dimensions<input class="input" value={dims()} onInput={(e) => setDims(e.currentTarget.value)} placeholder="height_cm=26, capacity_oz=12" /></label>
      <label>Features<input class="input" value={features()} onInput={(e) => setFeatures(e.currentTarget.value)} placeholder="leak-proof lid, dishwasher safe" /></label>
      <div class="row">
        <label>Beam width<input class="input" type="number" min="1" max="32" value={beam()} onInput={(e) => setBeam(Number(e.currentTarget.value) || 4)} /></label>
        <label>Max length<input class="input" type="number" min="16" max="512" value={maxLen()} onInput={(e) => setMaxLen(Number(e.currentTarget.value) || 192)} /></label>
      </div>
      <button class="btn btn-primary" disabled={props.pending}>
        {props.pending ? 'Generating…' : 'Generate drafts'}
      </button>
    </form>
  );
}
