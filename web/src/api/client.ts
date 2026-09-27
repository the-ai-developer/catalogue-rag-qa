/**
 * Typed HTTP client for every endpoint in docs/api-contract.md §1 (Go API).
 *
 * - Base URL: `VITE_API_BASE` + `/api/v1` (VITE_API_BASE may be empty for same-origin).
 * - Auth: `X-API-Key` header sourced from localStorage.
 * - Errors: any non-2xx is parsed from the normative error envelope and thrown as
 *   `ApiError` (code, message, details, HTTP status).
 * - Pagination: `?limit&cursor` helpers around `Page<T>`.
 */
import type {
  AnswerRecord,
  ApiErrorEnvelope,
  AskRequest,
  AskResponse,
  Asset,
  DescriptionJob,
  DescriptionJobRequest,
  Draft,
  EditorReview,
  EnqueueResponse,
  ErrorCode,
  GenerationJobStatus,
  IngestJob,
  Item,
  ItemInput,
  ItemListParams,
  ItemPatch,
  JobListParams,
  Page,
  PageParams,
  PublishedDescription,
  QaHistoryEntry,
  QaHistoryParams,
  ReviewRequest,
  SpecSheet,
} from './types';

// ---------------------------------------------------------------------------
// Configuration & auth
// ---------------------------------------------------------------------------

const API_KEY_STORAGE = 'catalogue-ai.api-key';

const RAW_BASE = (import.meta.env?.VITE_API_BASE ?? '') as string;
/** e.g. `http://localhost:8080/api/v1` or `/api/v1` behind a proxy. */
export const API_BASE = `${RAW_BASE.replace(/\/+$/, '')}/api/v1`;

export function getApiKey(): string {
  // Must agree with apiKey(): setApiKey falls back to an in-memory key when
  // localStorage is unavailable, so reading only storage here would report
  // "signed out" in exactly the environments where the in-memory fallback
  // exists. App.tsx uses this to decide whether to show the login screen.
  try {
    return localStorage.getItem(API_KEY_STORAGE) ?? memoryKey;
  } catch {
    return memoryKey;
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) localStorage.setItem(API_KEY_STORAGE, key);
    else localStorage.removeItem(API_KEY_STORAGE);
  } catch {
    /* storage unavailable (SSR/tests) — auth then relies on the in-memory key */
  }
  memoryKey = key;
}

export function clearApiKey(): void {
  setApiKey('');
}

let memoryKey = '';

function apiKey(): string {
  return getApiKey();
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  readonly code: ErrorCode;
  readonly status: number;
  readonly details: Record<string, unknown> | undefined;

  constructor(
    code: ErrorCode,
    message: string,
    status: number,
    details?: Record<string, unknown>,
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
  }

  /** Human-friendly short label, e.g. "Not found". */
  get codeLabel(): string {
    return this.code.replace(/_/g, ' ');
  }
}

const KNOWN_CODES: ReadonlySet<string> = new Set([
  'bad_request',
  'unauthorized',
  'forbidden',
  'not_found',
  'conflict',
  'unprocessable',
  'upstream',
  'internal',
]);

/**
 * Parse a non-2xx HTTP response into an `ApiError`.
 * Exported for unit tests; `request()` calls it on every failure.
 */
export async function parseErrorResponse(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    const text = await res.text();
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }

  const envelope = body as Partial<ApiErrorEnvelope> | null;
  const err = envelope?.error;
  if (err && typeof err === 'object' && typeof err.code === 'string' && KNOWN_CODES.has(err.code)) {
    return new ApiError(
      err.code as ErrorCode,
      typeof err.message === 'string' && err.message ? err.message : `HTTP ${res.status}`,
      res.status,
      err.details,
    );
  }

  // Non-conforming body: fall back to a code derived from the HTTP status.
  const code: ErrorCode =
    res.status === 400
      ? 'bad_request'
      : res.status === 401
        ? 'unauthorized'
        : res.status === 403
          ? 'forbidden'
          : res.status === 404
            ? 'not_found'
            : res.status === 409
              ? 'conflict'
              : res.status === 422
                ? 'unprocessable'
                : res.status >= 500
                  ? 'internal'
                  : 'internal';
  return new ApiError(code, `HTTP ${res.status} ${res.statusText || 'error'}`.trim(), res.status);
}

/** Normalise any thrown value into `ApiError` (for UI error banners). */
export function toApiError(e: unknown): ApiError {
  if (e instanceof ApiError) return e;
  if (e instanceof Error) return new ApiError('internal', e.message, 0);
  return new ApiError('internal', String(e), 0);
}

// ---------------------------------------------------------------------------
// Core request plumbing
// ---------------------------------------------------------------------------

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE';
  /** JSON body (mutually exclusive with `form`). */
  body?: unknown;
  /** Multipart body. */
  form?: FormData;
  query?: Record<string, string | number | boolean | undefined | null>;
  signal?: AbortSignal;
}

function buildUrl(path: string, query?: RequestOptions['query']): string {
  const url = `${API_BASE}${path}`;
  if (!query) return url;
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v !== undefined && v !== null && v !== '') params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' };
  const key = apiKey();
  if (key) headers['X-API-Key'] = key;

  let body: BodyInit | undefined;
  if (opts.form) {
    body = opts.form; // browser sets multipart boundary
  } else if (opts.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(opts.body);
  }

  let res: Response;
  try {
    res = await fetch(buildUrl(path, opts.query), {
      method: opts.method ?? 'GET',
      headers,
      body,
      signal: opts.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new ApiError('upstream', e instanceof Error ? e.message : 'network error', 0);
  }

  if (!res.ok) throw await parseErrorResponse(res);
  if (res.status === 204) return undefined as T;

  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

// ---------------------------------------------------------------------------
// Pagination helper
// ---------------------------------------------------------------------------

function pageQuery(params?: PageParams): Record<string, string | number | undefined> {
  return { limit: params?.limit, cursor: params?.cursor };
}

// ---------------------------------------------------------------------------
// 1.1 Items
// ---------------------------------------------------------------------------

export function createItem(input: ItemInput, signal?: AbortSignal): Promise<Item> {
  return request<Item>('/items', { method: 'POST', body: input, signal });
}

export function listItems(params?: ItemListParams, signal?: AbortSignal): Promise<Page<Item>> {
  return request<Page<Item>>('/items', {
    query: {
      ...pageQuery(params),
      query: params?.query,
      category: params?.category,
      status: params?.status,
    },
    signal,
  });
}

export function getItem(id: string, signal?: AbortSignal): Promise<Item> {
  return request<Item>(`/items/${encodeURIComponent(id)}`, { signal });
}

export function patchItem(id: string, patch: ItemPatch, signal?: AbortSignal): Promise<Item> {
  return request<Item>(`/items/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: patch,
    signal,
  });
}

export function deleteItem(id: string, signal?: AbortSignal): Promise<void> {
  return request<void>(`/items/${encodeURIComponent(id)}`, { method: 'DELETE', signal });
}

/** POST /items/{id}/assets — multipart `file` (png/jpg; the api sniffs the
 *  *  bytes and rejects anything that is not a real image). */
export function uploadAsset(id: string, file: File, signal?: AbortSignal): Promise<Asset> {
  const form = new FormData();
  form.append('file', file, file.name);
  return request<Asset>(`/items/${encodeURIComponent(id)}/assets`, {
    method: 'POST',
    form,
    signal,
  });
}

// ---------------------------------------------------------------------------
// 1.2 Ingest
// ---------------------------------------------------------------------------

export function enqueueIngest(id: string, signal?: AbortSignal): Promise<EnqueueResponse> {
  return request<EnqueueResponse>(`/items/${encodeURIComponent(id)}/ingest`, {
    method: 'POST',
    signal,
  });
}

export function getIngestJob(jobId: string, signal?: AbortSignal): Promise<IngestJob> {
  return request<IngestJob>(`/jobs/ingest/${encodeURIComponent(jobId)}`, { signal });
}

// ---------------------------------------------------------------------------
// 1.3 QA
// ---------------------------------------------------------------------------

export function ask(req: AskRequest, signal?: AbortSignal): Promise<AskResponse> {
  return request<AskResponse>('/qa/ask', { method: 'POST', body: req, signal });
}

export function getAnswer(answerId: string, signal?: AbortSignal): Promise<AnswerRecord> {
  return request<AnswerRecord>(`/qa/answers/${encodeURIComponent(answerId)}`, { signal });
}

export function getQaHistory(
  params?: QaHistoryParams,
  signal?: AbortSignal,
): Promise<Page<QaHistoryEntry>> {
  return request<Page<QaHistoryEntry>>('/qa/history', {
    query: { ...pageQuery(params), user_ref: params?.user_ref },
    signal,
  });
}

// ---------------------------------------------------------------------------
// 1.4 Descriptions
// ---------------------------------------------------------------------------

export function createDescriptionJob(
  req: DescriptionJobRequest,
  signal?: AbortSignal,
): Promise<{ job_id: string; status: 'queued' }> {
  return request<{ job_id: string; status: 'queued' }>('/descriptions/jobs', {
    method: 'POST',
    body: req,
    signal,
  });
}

export function getDescriptionJob(id: string, signal?: AbortSignal): Promise<DescriptionJob> {
  return request<DescriptionJob>(`/descriptions/jobs/${encodeURIComponent(id)}`, { signal });
}

export function listDescriptionJobs(
  params?: JobListParams,
  signal?: AbortSignal,
): Promise<Page<DescriptionJob>> {
  return request<Page<DescriptionJob>>('/descriptions/jobs', {
    query: { ...pageQuery(params), status: params?.status as GenerationJobStatus | undefined },
    signal,
  });
}

export function reviewDescriptionJob(
  id: string,
  req: ReviewRequest,
  signal?: AbortSignal,
): Promise<DescriptionJob> {
  return request<DescriptionJob>(`/descriptions/jobs/${encodeURIComponent(id)}/review`, {
    method: 'POST',
    body: req,
    signal,
  });
}

export function publishDescriptionJob(
  id: string,
  signal?: AbortSignal,
): Promise<PublishedDescription> {
  return request<PublishedDescription>(`/descriptions/jobs/${encodeURIComponent(id)}/publish`, {
    method: 'POST',
    signal,
  });
}

export function getPublishedDescriptions(
  itemId: string,
  params?: PageParams,
  signal?: AbortSignal,
): Promise<Page<PublishedDescription & { text: string }>> {
  return request<Page<PublishedDescription>>(`/descriptions/published/${encodeURIComponent(itemId)}`, {
    query: pageQuery(params),
    signal,
  });
}

// Re-export types used by consumers of this module.
export type {
  AnswerRecord,
  AskRequest,
  AskResponse,
  Asset,
  DescriptionJob,
  DescriptionJobRequest,
  Draft,
  EditorReview,
  EnqueueResponse,
  ErrorCode,
  IngestJob,
  Item,
  ItemInput,
  ItemListParams,
  ItemPatch,
  Page,
  PageParams,
  PublishedDescription,
  QaHistoryEntry,
  QaHistoryParams,
  ReviewRequest,
  SpecSheet,
};
