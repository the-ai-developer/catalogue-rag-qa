import { For, Show, createSignal, onMount } from 'solid-js';
import { ask, getQaHistory, toApiError } from '../api/client';
import type { AskResponse, QaHistoryEntry } from '../api/types';
import { CitationList, ErrorBanner, LoadingSkeleton } from '../components/ui';

/**
 * Project 1 — Multimodal RAG Catalogue Q&A.
 * Product rule: an answer is only displayed when its citation check passed.
 */
export default function AskPage() {
  const [question, setQuestion] = createSignal('');
  const [topK, setTopK] = createSignal(6);
  const [mode, setMode] = createSignal<'extractive' | 'abstractive'>('extractive');
  const [useImages, setUseImages] = createSignal(true);
  const [result, setResult] = createSignal<AskResponse | null>(null);
  const [history, setHistory] = createSignal<QaHistoryEntry[]>([]);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  onMount(async () => {
    try {
      setHistory((await getQaHistory({ limit: 10 })).items);
    } catch { /* history is optional */ }
  });

  const submit = async () => {
    if (!question().trim()) {
      setError('Ask a question about the catalogue.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await ask({
        question: question().trim(),
        top_k: topK(),
        use_images: useImages(),
        composition: mode(),
        user_ref: 'web:buyer',
      });
      setResult(res);
      setHistory((await getQaHistory({ limit: 10 })).items);
    } catch (e) {
      setError(toApiError(e).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section>
      <h2 class="page-title">Ask the catalogue</h2>
      <p class="muted">Answers are composed strictly from retrieved evidence (text + product photos) and every sentence must pass the citation check before display.</p>
      <ErrorBanner message={error()} onDismiss={() => setError(null)} />
      <form class="card form" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <label>
          Question
          <textarea class="input" rows={3} value={question()}
            onInput={(e) => setQuestion(e.currentTarget.value)}
            placeholder="Is the Kestrel bottle dishwasher safe and what is its capacity?" />
        </label>
        <div class="row">
          <label>Top-k
            <input class="input" type="number" min="1" max="20" value={topK()}
              onInput={(e) => setTopK(Number(e.currentTarget.value) || 6)} />
          </label>
          <label>Composition
            <select class="input" value={mode()} onChange={(e) => setMode(e.currentTarget.value as 'extractive' | 'abstractive')}>
              <option value="extractive">extractive (quote evidence)</option>
              <option value="abstractive">abstractive (draft + citation check)</option>
            </select>
          </label>
          <label>Images
            <input type="checkbox" checked={useImages()}
              onChange={(e) => setUseImages(e.currentTarget.checked)} /> search product photos too
          </label>
        </div>
        <button class="btn btn-primary" disabled={loading()}>
          {loading() ? 'Searching…' : 'Ask'}
        </button>
      </form>

      <Show when={!loading()} fallback={<LoadingSkeleton label="Retrieving evidence…" />}>
        <Show when={result()}>
          {(res) => (
            <div class="card">
              <Show when={res().retrieval.degraded}>
                {(d) => (
                  <div class="banner banner-warn" role="alert">
                    <strong>Retrieval is degraded.</strong>
                    <p>{d().detail} ({d().reason})</p>
                  </div>
                )}
              </Show>
              <Show
                when={res().citation_check.passed}
                fallback={
                  <div class="banner banner-warn" role="alert">
                    <strong>Ungrounded answer withheld.</strong>
                    <p>The citation check failed (unsupported sentences: {
                      res().citation_check.unsourced_sentences.join(', ') || 'n/a'
                    }). Try rephrasing or narrowing the scope.</p>
                  </div>
                }>
                <h3>Answer</h3>
                <p class="answer">{res().answer}</p>
              </Show>
              <p class="muted small">
                {res().composition} · {res().latency_ms} ms · retrieval:
                {' '}{res().retrieval.text_hits} text + {res().retrieval.image_hits} image hits
                · models: {res().retrieval.models.text}, {res().retrieval.models.image}
              </p>
              <CitationList citations={res().citations} passed={res().citation_check.passed} />
            </div>
          )}
        </Show>
      </Show>

      <h3>History</h3>
      <table class="quotes">
        <thead><tr><th>Question</th><th>Answer</th><th>Grounded</th></tr></thead>
        <tbody>
          <For each={history()} fallback={<tr><td colspan="3" class="muted">No questions asked yet.</td></tr>}>
            {(h) => (
              <tr>
                <td>{h.question || '—'}</td>
                <td>{h.citation_check_passed ? h.answer.slice(0, 80) + '…' : '—'}</td>
                <td>{h.citation_check_passed ? '✅' : 'withheld'}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </section>
  );
}
