/**
 * API types mirroring docs/api-contract.md §1 (Go API) JSON shapes exactly.
 * Where the contract does not pin a response body (e.g. GET /qa/answers/{id},
 * GET /qa/history, GET /descriptions/published/{item_id}), the shapes below are
 * the minimal consistent inference and are flagged in the project report.
 */

// ---------------------------------------------------------------------------
// Error envelope (any non-2xx)
// ---------------------------------------------------------------------------

export type ErrorCode =
  | 'bad_request'
  | 'unauthorized'
  | 'forbidden'
  | 'not_found'
  | 'conflict'
  | 'unprocessable'
  | 'upstream'
  | 'internal';

export interface ApiErrorEnvelope {
  error: {
    code: ErrorCode;
    message: string;
    details?: Record<string, unknown>;
  };
}

// ---------------------------------------------------------------------------
// Pagination convention
// ---------------------------------------------------------------------------

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
}

export interface PageParams {
  limit?: number; // 1..100, default 20
  cursor?: string;
}

// ---------------------------------------------------------------------------
// 1.1 Items
// ---------------------------------------------------------------------------

export type ItemStatus = 'draft' | 'active' | 'archived';

/** Free-form numeric dimension map, e.g. { height_cm: 26.0, capacity_oz: 12 }. */
export type Dimensions = Record<string, number>;

export interface Asset {
  id: string;
  kind: 'image' | 'document';
  url: string;
  mime: string;
  width: number | null;
  height: number | null;
  bytes: number;
  sha256: string;
  created_at: string;
}

export interface ItemCounts {
  chunks: number;
  embeddings_text: number;
  embeddings_image: number;
}

export interface Item {
  id: string;
  sku: string;
  title: string;
  category: string;
  material: string | null;
  dimensions: Dimensions;
  features: string[];
  extra: Record<string, unknown>;
  status: ItemStatus;
  created_at: string;
  updated_at: string;
  assets: Asset[];
  counts: ItemCounts;
}

/** POST /api/v1/items request body. */
export interface ItemInput {
  sku: string;
  title: string;
  category: string;
  material?: string | null;
  dimensions?: Dimensions;
  features?: string[];
  extra?: Record<string, unknown>;
  status?: ItemStatus;
}

/** PATCH /api/v1/items/{id} — any subset of these fields. */
export type ItemPatch = Partial<Pick<ItemInput, 'title' | 'category' | 'material' | 'dimensions' | 'features' | 'extra' | 'status'>>;

export interface ItemListParams extends PageParams {
  query?: string;
  category?: string;
  status?: ItemStatus;
}

// ---------------------------------------------------------------------------
// 1.2 Ingest
// ---------------------------------------------------------------------------

export type IngestJobStatus = 'queued' | 'running' | 'succeeded' | 'failed';

export interface IngestJob {
  id: string;
  item_id: string;
  status: IngestJobStatus;
  chunks_indexed: number;
  images_indexed: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

/** POST /api/v1/items/{id}/ingest → 202 */
export interface EnqueueResponse {
  job_id: string;
  status: 'queued';
}

// ---------------------------------------------------------------------------
// 1.3 QA (Project 1)
// ---------------------------------------------------------------------------

export type CompositionMode = 'extractive' | 'abstractive';
export type Modality = 'text' | 'image';

export interface AskRequest {
  question: string;
  item_ids?: string[]; // empty/absent = whole catalogue
  top_k?: number;
  use_images?: boolean;
  composition?: CompositionMode; // default extractive
  user_ref?: string;
}

export interface Citation {
  sentence_index: number;
  item_id: string;
  sku: string;
  chunk_id: string | null;
  asset_id: string | null;
  modality: Modality;
  score: number;
  snippet: string;
}

export interface CitationCheck {
  passed: boolean;
  unsourced_sentences: number[];
}

/** Degradation report from the model server, when the shared space is unusable. */
export interface Degraded {
  reason: string;
  detail: string;
  path?: string;
}

export interface RetrievalInfo {
  text_hits: number;
  image_hits: number;
  models: { text: string; image: string };
  /** Present when the model server is running without trained projection
   *  weights: cross-modal scores cannot be trusted. */
  degraded?: Degraded | null;
}

export interface AskResponse {
  answer_id: string;
  answer: string;
  composition: CompositionMode;
  citations: Citation[];
  citation_check: CitationCheck;
  retrieval: RetrievalInfo;
  latency_ms: number;
}

/** GET /api/v1/qa/answers/{id} — stored answer + citations (query metadata joined). */
export interface AnswerRecord {
  answer_id: string;
  question: string;
  user_ref: string;
  answer: string;
  composition: CompositionMode;
  citations: Citation[];
  citation_check: CitationCheck;
  retrieval: RetrievalInfo;
  latency_ms: number;
  created_at: string;
}

/** GET /api/v1/qa/history — recent Q&A, optional ?user_ref= filter. */
export interface QaHistoryEntry {
  answer_id: string;
  query_id: string;
  /** Joined from qa_queries; the api used to omit it, blanking the history table. */
  question: string;
  user_ref: string;
  answer: string;
  composition: CompositionMode;
  model_version: string;
  citation_check: CitationCheck;
  citation_check_passed: boolean;
  latency_ms: number;
  created_at: string;
}

export interface QaHistoryParams extends PageParams {
  user_ref?: string;
}

// ---------------------------------------------------------------------------
// 1.4 Descriptions (Project 2)
// ---------------------------------------------------------------------------

export interface SpecSheet {
  item_id?: string | null;
  title?: string;
  category: string;
  material?: string;
  dimensions?: Dimensions;
  features?: string[];
  extra?: Record<string, unknown>;
}

export interface DescriptionJobRequest {
  spec: SpecSheet;
  beam_width?: number;
  max_len?: number;
}

export type GenerationJobStatus =
  | 'queued'
  | 'running'
  | 'draft_ready'
  | 'approved'
  | 'rejected'
  | 'published'
  | 'failed';

export interface Draft {
  id: string;
  rank: number;
  text: string;
  score: number;
}

export type ReviewDecision = 'approve' | 'reject' | 'edit';

export interface EditorReview {
  id: string;
  draft_id: string | null;
  reviewer: string;
  decision: ReviewDecision;
  edited_text: string | null;
  notes: string | null;
  created_at: string;
}

export interface PublishedDescription {
  id: string;
  item_id: string;
  text: string;
  approved_by: string;
  created_at?: string;
}

export interface DescriptionJob {
  id: string;
  spec: SpecSheet;
  status: GenerationJobStatus;
  model_name: string | null;
  model_version: string | null;
  beam_width: number;
  max_len: number;
  drafts: Draft[];
  reviews: EditorReview[];
  published: PublishedDescription[];
  error?: string | null;
  created_at?: string;
}

export interface ReviewRequest {
  decision: ReviewDecision;
  draft_id?: string; // required for approve/edit
  edited_text?: string; // required for edit
  notes?: string;
}

export interface JobListParams extends PageParams {
  status?: GenerationJobStatus;
}
