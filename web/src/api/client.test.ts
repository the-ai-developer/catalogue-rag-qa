import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  API_BASE, ApiError, clearApiKey, getApiKey, parseErrorResponse, setApiKey,
  toApiError,
} from './client';

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('API_BASE', () => {
  it('targets the same-origin /api/v1 when no base is configured', () => {
    // The SPA is built with an empty VITE_API_BASE and nginx reverse-proxies
    // /api to the Go service, so this is the production shape.
    expect(API_BASE).toMatch(/\/api\/v1$/);
  });
});

describe('parseErrorResponse', () => {
  it('unwraps the normative error envelope', async () => {
    const err = await parseErrorResponse(
      json({ error: { code: 'not_found', message: 'item not found', details: { id: 'x' } } }, 404),
    );
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe('not_found');
    expect(err.message).toBe('item not found');
    expect(err.status).toBe(404);
    expect(err.details).toEqual({ id: 'x' });
  });

  it('derives a code from the status when the body is not the envelope', async () => {
    // A proxy 502 or an nginx HTML error page must still surface as an ApiError
    // rather than throwing a JSON parse error at the UI.
    const cases: [number, string][] = [
      [400, 'bad_request'], [401, 'unauthorized'], [403, 'forbidden'],
      [404, 'not_found'], [409, 'conflict'], [422, 'unprocessable'],
      [500, 'internal'], [502, 'internal'], [503, 'internal'],
    ];
    for (const [status, code] of cases) {
      const err = await parseErrorResponse(new Response('<html>oops</html>', { status }));
      expect(err.code, `status ${status}`).toBe(code);
      expect(err.status).toBe(status);
    }
  });

  it('survives an empty body', async () => {
    const err = await parseErrorResponse(new Response('', { status: 401 }));
    expect(err.code).toBe('unauthorized');
  });

  it('rejects an unknown code in the envelope and falls back to the status', async () => {
    // A code we do not know about must not leak into the UI as-is.
    const err = await parseErrorResponse(
      json({ error: { code: 'teapot', message: 'nope' } }, 418),
    );
    expect(err.code).toBe('internal');
  });

  it('ignores a non-string code', async () => {
    const err = await parseErrorResponse(json({ error: { code: 42, message: 'x' } }, 400));
    expect(err.code).toBe('bad_request');
  });

  it('exposes a human label', async () => {
    const err = await parseErrorResponse(
      json({ error: { code: 'unprocessable', message: 'm' } }, 422),
    );
    expect(err.codeLabel).toBe('unprocessable');
  });
});

describe('toApiError', () => {
  it('passes ApiError through untouched', () => {
    const e = new ApiError('conflict', 'sku already exists', 409);
    expect(toApiError(e)).toBe(e);
  });

  it('wraps a plain Error as internal with status 0', () => {
    const e = toApiError(new Error('boom'));
    expect(e.code).toBe('internal');
    expect(e.status).toBe(0);
    expect(e.message).toBe('boom');
  });

  it('stringifies a non-Error throw', () => {
    // fetch rejections and library code can throw anything.
    expect(toApiError('a string').message).toBe('a string');
    expect(toApiError(undefined).message).toBe('undefined');
  });
});

describe('api key storage', () => {
  beforeEach(() => clearApiKey());
  afterEach(() => clearApiKey());

  it('round-trips a key', () => {
    setApiKey('local-dev-admin-key');
    expect(getApiKey()).toBe('local-dev-admin-key');
  });

  it('clears the key', () => {
    setApiKey('k');
    clearApiKey();
    expect(getApiKey()).toBe('');
  });
});
