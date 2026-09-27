/** App shell: sidebar navigation + nested routed content. */
import { For, Show, createSignal, onMount, type JSX } from 'solid-js';
import { A, useNavigate } from '@solidjs/router';
import { clearApiKey, getApiKey, getQaHistory, listDescriptionJobs, listItems } from './api/client';
import type { DescriptionJob, Item, QaHistoryEntry } from './api/types';
import { StatusBadge } from './components/ui';

export default function App(props: { children?: JSX.Element }) {
  const navigate = useNavigate();
  const [authed, setAuthed] = createSignal(!!getApiKey());
  const [items, setItems] = createSignal<Item[]>([]);
  const [jobs, setJobs] = createSignal<DescriptionJob[]>([]);
  const [history, setHistory] = createSignal<QaHistoryEntry[]>([]);

  onMount(async () => {
    if (!authed()) {
      navigate('/login');
      return;
    }
    try {
      setItems((await listItems({ limit: 5 })).items);
      setJobs((await listDescriptionJobs({ limit: 5 })).items);
      setHistory((await getQaHistory({ limit: 5 })).items);
    } catch {
      /* sidebar panels are best-effort */
    }
  });

  return (
    <div class="layout">
      <aside class="sidebar">
        <h1><A href="/">Catalogue AI</A></h1>
        <nav>
          <A href="/catalogue">Catalogue</A>
          <A href="/ask">Ask (RAG)</A>
          <A href="/descriptions">Descriptions</A>
        </nav>
        <Show when={authed()} fallback={
          <button class="btn" onClick={() => navigate('/login')}>Set API key</button>
        }>
          <button class="btn" onClick={() => { clearApiKey(); setAuthed(false); navigate('/login'); }}>
            Sign out
          </button>
        </Show>
        <section class="sidebar-section">
          <h2>Items</h2>
          <For each={items()} fallback={<p class="muted small">No items yet.</p>}>
            {(it) => (
              <div class="sidebar-row">
                <A href={`/catalogue/${it.id}`}>{it.title}</A>
                <StatusBadge status={it.status} />
              </div>
            )}
          </For>
        </section>
        <section class="sidebar-section">
          <h2>Description jobs</h2>
          <For each={jobs()} fallback={<p class="muted small">No jobs yet.</p>}>
            {(j) => (
              <div class="sidebar-row">
                <A href={`/descriptions?job=${j.id}`}>#{j.id.slice(0, 8)}</A>
                <StatusBadge status={j.status} />
              </div>
            )}
          </For>
        </section>
        <section class="sidebar-section">
          <h2>Recent Q&amp;A</h2>
          <For each={history()} fallback={<p class="muted small">No questions yet.</p>}>
            {(h) => (
              <div class="sidebar-row">
                <span class="small">{(h.question || '—').slice(0, 28)}{h.question ? '…' : ''}</span>
                <StatusBadge status={h.citation_check_passed ? 'ok' : 'withheld'} />
              </div>
            )}
          </For>
        </section>
      </aside>
      <main class="content">
        {props.children}
      </main>
    </div>
  );
}
