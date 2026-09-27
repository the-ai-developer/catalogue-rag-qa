import { For, Show, createSignal, onMount } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { createItem, listItems } from '../api/client';
import type { Item } from '../api/types';
import { ErrorBanner, ItemCard, LoadingSkeleton, SearchBar } from '../components/ui';
import { toApiError } from '../api/client';

export default function CataloguePage() {
  const navigate = useNavigate();
  const [items, setItems] = createSignal<Item[]>([]);
  const [query, setQuery] = createSignal('');
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [showForm, setShowForm] = createSignal(false);
  const [title, setTitle] = createSignal('');
  const [sku, setSku] = createSignal('');
  const [category, setCategory] = createSignal('');
  const [pending, setPending] = createSignal(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setItems((await listItems({ query: query() || undefined, limit: 50 })).items);
      setError(null);
    } catch (e) {
      setError(toApiError(e).message);
    } finally {
      setLoading(false);
    }
  };

  onMount(refresh);

  const submit = async () => {
    if (!title().trim() || !sku().trim() || !category().trim()) {
      setError('sku, title and category are required');
      return;
    }
    setPending(true);
    try {
      await createItem({ sku: sku().trim(), title: title().trim(), category: category().trim(), status: 'draft' });
      setTitle(''); setSku(''); setCategory(''); setShowForm(false);
      await refresh();
    } catch (e) {
      setError(toApiError(e).message);
    } finally {
      setPending(false);
    }
  };

  return (
    <section>
      <h2 class="page-title">Catalogue</h2>
      <ErrorBanner message={error()} onDismiss={() => setError(null)} />
      <div class="row">
        <SearchBar value={query()} onInput={(v) => { setQuery(v); refresh(); }} />
        <button class="btn btn-primary" onClick={() => setShowForm(!showForm())}>+ New item</button>
      </div>
      <Show when={showForm()}>
        <form class="card form" onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <label>SKU<input class="input" value={sku()} onInput={(e) => setSku(e.currentTarget.value)} placeholder="KX-1042" /></label>
          <label>Title<input class="input" value={title()} onInput={(e) => setTitle(e.currentTarget.value)} placeholder="Kestrel 12 oz Insulated Bottle" /></label>
          <label>Category<input class="input" value={category()} onInput={(e) => setCategory(e.currentTarget.value)} placeholder="drinkware" /></label>
          <button class="btn btn-primary" disabled={pending()}>{pending() ? 'Creating…' : 'Create item'}</button>
        </form>
      </Show>
      <Show when={!loading()} fallback={<LoadingSkeleton label="Loading catalogue…" />}>
        <Show when={items().length} fallback={<p class="muted">No items yet — create one above.</p>}>
          <div class="grid">
            <For each={items()}>{(it) => <ItemCard item={it} onOpen={(id) => navigate(`/catalogue/${id}`)} />}</For>
          </div>
        </Show>
      </Show>
    </section>
  );
}
