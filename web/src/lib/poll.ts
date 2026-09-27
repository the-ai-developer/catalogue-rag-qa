/**
 * Polling helper for job status.
 *
 * Polls `fetcher` while the predicate says the job is pending; stops on a
 * terminal state, on an owner `stop()`, or on component cleanup. Every tick
 * replaces `value`.
 *
 * The `inFlight` guard is the point of using this: a hand-rolled `setInterval`
 * fires a new request every tick regardless of whether the previous one
 * finished, so a slow endpoint (an ingest that takes 8s, a 2s interval) queues
 * up overlapping requests and renders stale responses out of order.
 */
import { createSignal, onCleanup, type Accessor } from 'solid-js';

export const POLL_INTERVAL_MS = 2000;

export interface PollController<T> {
  /** Latest polled value (undefined until the first tick). */
  value: Accessor<T | undefined>;
  /** True while a tick is in flight. */
  pending: Accessor<boolean>;
  stop: () => void;
}

export interface PollOptions<T> {
  /** Called on every successful tick, including the first. */
  onValue?: (value: T) => void;
  /** Called when a tick throws. Returning false stops the poll. */
  onError?: (error: unknown) => boolean | void;
}

/**
 * Start polling `fetcher` until `isTerminal(latest)` is true.
 * The first fetch happens immediately; subsequent ticks are scheduled only
 * after the previous one settles.
 */
export function pollUntil<T>(
  fetcher: () => Promise<T>,
  isTerminal: (value: T) => boolean,
  intervalMs: number = POLL_INTERVAL_MS,
  options: PollOptions<T> = {},
): PollController<T> {
  const { onValue, onError } = options;
  const [value, setValue] = createSignal<T>();
  const [pending, setPending] = createSignal(false);
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let inFlight = false;

  const tick = async () => {
    if (stopped || inFlight) return;
    inFlight = true;
    setPending(true);
    try {
      const next = await fetcher();
      if (stopped) return;
      setValue(() => next);
      onValue?.(next);
      if (isTerminal(next)) {
        stopped = true;
        return;
      }
    } catch (e) {
      // Transient fetch errors: keep polling unless the caller opts out, so a
      // single 502 during a rollout does not abandon a running job.
      if (onError?.(e) === false) stopped = true;
    } finally {
      inFlight = false;
      setPending(false);
    }
    if (!stopped) timer = setTimeout(tick, intervalMs);
  };

  const stop = () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    timer = undefined;
  };

  onCleanup(stop);
  void tick();
  return { value, pending, stop };
}

/** Job status helpers shared by ingest + description jobs. */
export const INGEST_TERMINAL = new Set(['succeeded', 'failed']);
export const DESC_TERMINAL = new Set([
  'draft_ready',
  'approved',
  'rejected',
  'published',
  'failed',
]);

export function isJobPending(status: string, terminal: Set<string>): boolean {
  return !terminal.has(status);
}
