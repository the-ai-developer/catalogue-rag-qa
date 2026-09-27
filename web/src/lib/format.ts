/** Small pure formatting helpers shared across pages/components. */

/** RFC 3339 → "2026-09-27" (local date). Returns "—" for missing/invalid input. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

/** RFC 3339 → "2026-09-27 03:44" (local date+time). */
export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Duration in ms → "820 ms" / "1.4 s" / "2 min 5 s". */
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const min = Math.floor(s / 60);
  const rem = Math.round(s - min * 60);
  return `${min} min ${rem} s`;
}

/** Bytes → "482 KB" etc. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
}

/** Integer with thousands separators. */
export function formatNumber(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return n.toLocaleString();
}

/** Similarity score → "0.83". */
export function formatScore(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return '—';
  return score.toFixed(2);
}

/** Truncate with ellipsis at word boundary when possible. */
export function truncate(text: string, max = 160): string {
  const t = text.trim();
  if (t.length <= max) return t;
  const cut = t.slice(0, max);
  const sp = cut.lastIndexOf(' ');
  return `${(sp > max * 0.6 ? cut.slice(0, sp) : cut).trimEnd()}…`;
}

/** Naive pluralisation for UI copy. */
export function plural(n: number, singular: string, pluralForm?: string): string {
  return `${formatNumber(n)} ${n === 1 ? singular : (pluralForm ?? `${singular}s`)}`;
}

/**
 * Render a jsonb object for display. Returns "—" for null/undefined/empty and
 * never throws: a stored JSON `null` (which older rows can carry) must not take
 * the page down with it.
 */
export function formatJSON(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object' && Object.keys(value as object).length === 0) return '—';
  try {
    const out = JSON.stringify(value);
    return out === '{}' || out === 'null' || out === undefined ? '—' : out;
  } catch {
    return '—';
  }
}

/** Human label for snake_case enum values: "draft_ready" → "Draft ready". */
export function humanize(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
