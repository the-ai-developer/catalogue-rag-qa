import { describe, expect, it } from 'vitest';
import {
  formatBytes, formatDate, formatDateTime, formatDuration, formatJSON, formatNumber,
  formatScore, humanize, plural, truncate,
} from './format';

const MISSING = '—';

describe('formatDate / formatDateTime', () => {
  it('renders a real timestamp', () => {
    expect(formatDate('2026-09-27T10:30:00Z')).toMatch(/2026/);
    expect(formatDateTime('2026-09-27T10:30:00Z')).toMatch(/2026/);
  });

  it('returns the em dash for missing or invalid input', () => {
    for (const bad of [null, undefined, '', 'not-a-date', '2026-13-45']) {
      expect(formatDate(bad)).toBe(MISSING);
      expect(formatDateTime(bad)).toBe(MISSING);
    }
  });
});

describe('formatDuration', () => {
  it('scales the unit', () => {
    expect(formatDuration(0)).toBe('0 ms');
    expect(formatDuration(820)).toBe('820 ms');
    expect(formatDuration(1400)).toBe('1.4 s');
    expect(formatDuration(59_000)).toBe('59.0 s');
    expect(formatDuration(125_000)).toBe('2 min 5 s');
  });

  it('returns the em dash for nullish/NaN', () => {
    expect(formatDuration(null)).toBe(MISSING);
    expect(formatDuration(undefined)).toBe(MISSING);
    expect(formatDuration(Number.NaN)).toBe(MISSING);
  });
});

describe('formatBytes', () => {
  it('scales by 1024', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(1024 * 1024)).toBe('1.0 MB');
    expect(formatBytes(1024 ** 3)).toBe('1.0 GB');
  });

  it('returns the em dash for nullish/NaN', () => {
    expect(formatBytes(null)).toBe(MISSING);
    expect(formatBytes(Number.NaN)).toBe(MISSING);
  });
});

describe('formatNumber / formatScore', () => {
  it('formats numbers and scores', () => {
    expect(formatNumber(1234)).toMatch(/1.234|1,234/);
    expect(formatNumber(0)).toBe('0');
    expect(formatScore(0.8312)).toBe('0.83');
    expect(formatScore(1)).toBe('1.00');
  });

  it('returns the em dash for nullish/NaN', () => {
    expect(formatNumber(null)).toBe(MISSING);
    expect(formatScore(undefined)).toBe(MISSING);
    expect(formatScore(Number.NaN)).toBe(MISSING);
  });
});

describe('truncate', () => {
  it('leaves short text alone', () => {
    expect(truncate('short')).toBe('short');
  });

  it('trims and truncates long text with an ellipsis', () => {
    const out = truncate('the quick brown fox jumps over the lazy dog again', 20);
    expect(out.endsWith('…')).toBe(true);
    expect(out.length).toBeLessThanOrEqual(21);
  });

  it('hard-cuts when there is no usable word boundary', () => {
    const out = truncate('x'.repeat(50), 10);
    expect(out).toBe(`${'x'.repeat(10)}…`);
  });
});

describe('plural', () => {
  it('switches on exactly one', () => {
    expect(plural(1, 'citation')).toMatch(/1 citation$/);
    expect(plural(0, 'citation')).toMatch(/0 citations$/);
    expect(plural(2, 'citation')).toMatch(/2 citations$/);
  });

  it('accepts an irregular plural', () => {
    expect(plural(2, 'entry', 'entries')).toMatch(/entries$/);
  });
});

describe('humanize', () => {
  it('turns snake_case into a label', () => {
    // These are the job statuses from the workflow state machine.
    expect(humanize('draft_ready')).toBe('Draft Ready');
    expect(humanize('approved')).toBe('Approved');
    expect(humanize('succeeded')).toBe('Succeeded');
  });

  it('leaves an empty value alone', () => {
    expect(humanize('')).toBe('');
  });
});

describe('formatJSON', () => {
  it('renders a populated object', () => {
    expect(formatJSON({ height_cm: 26 })).toBe('{"height_cm":26}');
  });

  it('renders the em dash for absent or empty values', () => {
    // A stored jsonb `null` (older rows carry one) must not break the page.
    for (const bad of [null, undefined, {}, { a: undefined }]) {
      expect(formatJSON(bad)).toBe(MISSING);
    }
  });

  it('never throws on a circular or exotic value', () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(formatJSON(circular)).toBe(MISSING);
    expect(formatJSON(42)).toBe('42');
  });
})
