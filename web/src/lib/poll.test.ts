import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createRoot } from 'solid-js';
import {
  DESC_TERMINAL, INGEST_TERMINAL, POLL_INTERVAL_MS, isJobPending, pollUntil,
} from './poll';

/** Poll outside a reactive root so onCleanup does not fire immediately. */
function run<T>(
  fetcher: () => Promise<T>,
  isTerminal: (v: T) => boolean,
  intervalMs = 1,
  options?: Parameters<typeof pollUntil<T>>[3],
) {
  let controller!: ReturnType<typeof pollUntil<T>>;
  const disposeRoot = createRoot((disposeFn) => {
    controller = pollUntil(fetcher, isTerminal, intervalMs, options);
    return disposeFn;
  });
  return { controller, disposeRoot };
}

const tick = (ms = 4) => new Promise((r) => setTimeout(r, ms));

describe('pollUntil', () => {
  it('fetches immediately rather than waiting a full interval', async () => {
    const fetcher = vi.fn().mockResolvedValue({ status: 'done' });
    const { controller, disposeRoot } = run(fetcher, () => true, 10_000);
    await tick();
    expect(fetcher).toHaveBeenCalledTimes(1);
    controller.stop();
    disposeRoot();
  });

  it('stops on a terminal value and reports it once', async () => {
    const values = [{ s: 'queued' }, { s: 'running' }, { s: 'succeeded' }];
    let i = 0;
    const onValue = vi.fn();
    const { controller, disposeRoot } = run(
      async () => values[Math.min(i++, values.length - 1)],
      (v) => v.s === 'succeeded',
      1,
      { onValue },
    );
    await tick(30);
    expect(onValue).toHaveBeenCalledTimes(3);
    expect(onValue.mock.calls.map((c) => c[0].s)).toEqual(['queued', 'running', 'succeeded']);
    expect(controller.value()?.s).toBe('succeeded');
    const callsAtStop = fetcherCalls(onValue);
    await tick(20);
    expect(fetcherCalls(onValue)).toBe(callsAtStop); // no polling after terminal
    disposeRoot();
  });

  it('does not overlap requests when the fetcher is slower than the interval', async () => {
    // The reason this helper exists: a hand-rolled setInterval fires a new
    // request every tick regardless of whether the last one finished.
    let inFlight = 0;
    let maxInFlight = 0;
    let done = false;
    const { disposeRoot } = run(
      async () => {
        inFlight++;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await tick(15);
        inFlight--;
        if (done) return { status: 'succeeded' };
        return { status: 'running' };
      },
      (v) => v.status === 'succeeded',
      1,
    );
    await tick(60);
    done = true;
    await tick(40);
    expect(maxInFlight).toBe(1);
    disposeRoot();
  });

  it('keeps polling after a transient error by default', async () => {
    let calls = 0;
    const onValue = vi.fn();
    const { disposeRoot } = run(
      async () => {
        calls++;
        if (calls === 1) throw new Error('502 from the model server');
        return { status: 'succeeded' };
      },
      (v) => v.status === 'succeeded',
      1,
      { onValue },
    );
    await tick(40);
    expect(calls).toBeGreaterThan(1);
    expect(onValue).toHaveBeenCalledWith({ status: 'succeeded' });
    disposeRoot();
  });

  it('stops when onError returns false', async () => {
    let calls = 0;
    const { disposeRoot } = run(
      async () => {
        calls++;
        throw new Error('404 job not found');
      },
      () => false,
      1,
      { onError: () => false },
    );
    await tick(30);
    const seen = calls;
    await tick(20);
    expect(calls).toBe(seen);
    disposeRoot();
  });

  it('stop() halts further polling', async () => {
    const fetcher = vi.fn().mockResolvedValue({ status: 'running' });
    const { controller, disposeRoot } = run(fetcher, () => false, 1);
    await tick(20);
    const before = fetcher.mock.calls.length;
    controller.stop();
    await tick(20);
    expect(fetcher.mock.calls.length).toBe(before);
    disposeRoot();
  });

  it('stops on owner disposal (component unmount)', async () => {
    const fetcher = vi.fn().mockResolvedValue({ status: 'running' });
    const disposeRoot = createRoot((disposeFn) => {
      pollUntil(fetcher, () => false, 1);
      return disposeFn;
    });
    await tick(20);
    const before = fetcher.mock.calls.length;
    disposeRoot();
    await tick(20);
    expect(fetcher.mock.calls.length).toBe(before);
  });

  it('reports in-flight state', async () => {
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    const { controller, disposeRoot } = run(
      async () => { await gate; return { status: 'running' }; },
      () => false,
      1,
    );
    await tick();
    expect(controller.pending()).toBe(true);
    release();
    await tick(10);
    expect(controller.pending()).toBe(false);
    controller.stop();
    disposeRoot();
  });
});

function fetcherCalls(onValue: ReturnType<typeof vi.fn>): number {
  return onValue.mock.calls.length;
}

describe('terminal status sets', () => {
  it('treat every documented ingest terminal state as terminal', () => {
    expect(INGEST_TERMINAL.has('succeeded')).toBe(true);
    expect(INGEST_TERMINAL.has('failed')).toBe(true);
    expect(INGEST_TERMINAL.has('running')).toBe(false);
    expect(isJobPending('running', INGEST_TERMINAL)).toBe(true);
    expect(isJobPending('succeeded', INGEST_TERMINAL)).toBe(false);
  });

  it('covers every description state the worker can produce', () => {
    // A status missing from this set means the UI polls forever.
    for (const s of ['draft_ready', 'approved', 'rejected', 'published', 'failed']) {
      expect(DESC_TERMINAL.has(s)).toBe(true);
    }
    for (const s of ['queued', 'running']) {
      expect(DESC_TERMINAL.has(s)).toBe(false);
    }
  });

  it('has a sane default interval', () => {
    expect(POLL_INTERVAL_MS).toBeGreaterThan(0);
    expect(POLL_INTERVAL_MS).toBeLessThanOrEqual(5000);
  });
});
