/**
 * Browser-level contract tests. See playwright.config.ts for how to run them.
 *
 * Every assertion here corresponds to a bug that reached main or a rule the
 * server alone cannot enforce.
 */
import { expect, test, type Page } from '@playwright/test';

const KEY = process.env.API_KEY ?? 'local-dev-admin-key';

async function signIn(page: Page) {
  // The SPA reads the key from localStorage on first paint, so it has to be in
  // place before any app code runs or the app renders the login screen.
  await page.addInitScript((key: string) => {
    window.localStorage.setItem('catalogue-ai.api-key', key);
  }, KEY);
}

/** Requests that never reached the Go service, i.e. the proxy is broken. */
function trackFailedApi(page: Page) {
  const failures: string[] = [];
  page.on('response', (res) => {
    if (res.url().includes('/api/') && res.status() >= 400) {
      failures.push(`${res.status()} ${res.request().method()} ${res.url()}`);
    }
  });
  return failures;
}

test.describe('shell', () => {
  test('serves the SPA and does not bounce a signed-in user to /login', async ({ page }) => {
    await signIn(page);
    await page.goto('/');
    // `/` is routed to the catalogue, so the shell and the catalogue both render
    // and the URL stays put.
    await expect(page.locator('h1')).toContainText('Catalogue AI');
    await expect(page.locator('h2.page-title')).toContainText('Catalogue');
    await expect(page).toHaveURL(/\/$|\/catalogue/);
  });

  test('sends an unauthenticated visitor to the login page', async ({ page }) => {
    // No key in localStorage: App's onMount must redirect, otherwise the user
    // stares at an empty shell and every request 401s.
    await page.goto('/catalogue');
    await expect(page).toHaveURL(/\/login/);
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('serves the SPA for a deep link (history fallback)', async ({ page }) => {
    await signIn(page);
    await page.goto('/descriptions');
    await expect(page).toHaveTitle(/Catalogue|Bifrost|vite/i);
    await expect(page.locator('h2.page-title')).toContainText('Spec');
  });

  test('proxies /api to the Go service', async ({ page }) => {
    const failures = trackFailedApi(page);
    await signIn(page);
    await page.goto('/catalogue');
    // The sidebar lists items from GET /api/v1/items; an unproxied /api would
    // 404 and the sidebar would stay empty.
    await expect(page.locator('section.sidebar-section').first()).toBeVisible();
    expect(failures, `api calls failed: ${failures.join(', ')}`).toHaveLength(0);
  });

  test('exposes /readyz as JSON through the same origin', async ({ request }) => {
    // Regression: /readyz fell through to the SPA fallback and returned HTML.
    const res = await request.get('/readyz');
    expect(res.status()).toBe(200);
    expect(res.headers()['content-type']).toContain('json');
    expect((await res.json()).ok).toBe(true);
  });
});

test.describe('catalogue', () => {
  test('renders items returned by the API', async ({ page }) => {
    const failures = trackFailedApi(page);
    await signIn(page);
    await page.goto('/catalogue');
    await expect(page.locator('table.quotes, .grid').first()).toBeVisible();
    expect(failures, `api calls failed: ${failures.join(', ')}`).toHaveLength(0);
  });

  test('creates an item and sees it in the list', async ({ page }) => {
    const failures = trackFailedApi(page);
    await signIn(page);
    await page.goto('/catalogue');

    await page.getByRole('button', { name: '+ New item' }).click();
    const sku = `E2E-${Date.now().toString(36).toUpperCase()}`;
    await page.locator('input[placeholder="KX-1042"]').fill(sku);
    await page.locator('input[placeholder*="Kestrel"]').fill('E2E Probe Item');
    await page.locator('input[placeholder="drinkware"]').fill('drinkware');
    await page.getByRole('button', { name: 'Create item' }).click();

    await expect(page.locator('body')).toContainText('E2E Probe Item');
    expect(failures, `api calls failed: ${failures.join(', ')}`).toHaveLength(0);
  });
});

test.describe('Project 1 — the citation gate is enforced in the UI', () => {
  test('never renders answer text when the citation check failed', async ({ page }) => {
    await signIn(page);
    // Force the failure path by asking something the catalogue cannot ground.
    // Whichever branch the server takes, the UI must withhold the answer.
    await page.goto('/ask');
    await page.locator('textarea').fill('What is the warranty on the lunar rover?');
    await page.getByRole('button', { name: 'Ask' }).click();

    // Either the refusal banner or a withheld answer is acceptable; the actual
    // answer prose must never appear.
    await expect(page.locator('.banner').first()).toBeVisible();
    await expect(page.locator('p.answer')).toHaveCount(0);
  });

  test('surfaces a degraded shared space instead of hiding it', async ({ page }) => {
    await signIn(page);
    await page.goto('/ask');
    await page.locator('textarea').fill('Which bottle is dishwasher safe?');
    await page.getByRole('button', { name: 'Ask' }).click();
    // With an untrained projection the server reports degraded; the page must
    // surface that rather than presenting noise as a result.
    const degraded = page.locator('.banner', { hasText: 'degraded' });
    const withheld = page.locator('.banner', { hasText: 'withheld' });
    await expect(degraded.or(withheld).first()).toBeVisible();
  });

  test('history table shows the question, not a blank cell', async ({ page }) => {
    // Regression: the API omitted `question` and tagged citation_check_passed as
    // json:"-", so every row rendered blank and every answer read "withheld".
    await signIn(page);
    await page.goto('/ask');
    await page.locator('textarea').fill('Which bottle is dishwasher safe?');
    await page.getByRole('button', { name: 'Ask' }).click();
    await expect(page.locator('table.quotes tbody tr').first()).toBeVisible();
    const firstCell = page.locator('table.quotes tbody tr').first().locator('td').first();
    await expect(firstCell).not.toHaveText('');
  });
});

test.describe('product photos', () => {
  test('serves an uploaded photo from the /assets volume, not the SPA', async ({ page, request }) => {
    // The regression this guards: Vite's bundle also lived under /assets/, which
    // nginx aliases to the photo volume, so the app's own JS 404'd and the page
    // rendered blank. The two URL spaces must stay disjoint.
    const failures = trackFailedApi(page);
    await signIn(page);
    await page.goto('/catalogue');

    // Find an item that actually has a photo, via the API the app already uses.
    const res = await request.get('/api/v1/items?limit=100', {
      headers: { 'X-API-Key': KEY },
    });
    expect(res.status()).toBe(200);
    const items = (await res.json()).items as Array<{ id: string; assets: Array<{ url: string }> }>;
    const withPhoto = items.find((i) => i.assets.length > 0);
    test.skip(!withPhoto, 'no item has a photo yet (run scripts/seed_demo.py --with-images)');

    const photo = await request.get(withPhoto!.assets[0].url);
    expect(photo.status()).toBe(200);
    // An SPA fallback would answer 200 text/html, which is how the collision
    // presented itself.
    expect(photo.headers()['content-type']).toMatch(/^image\//);

    // …and the same photo renders on the item page.
    await page.goto(`/catalogue/${withPhoto!.id}`);
    const img = page.locator('figure img').first();
    await expect(img).toBeVisible();
    expect(await img.evaluate((el: HTMLImageElement) => el.naturalWidth)).toBeGreaterThan(0);
    expect(failures, `api calls failed: ${failures.join(', ')}`).toHaveLength(0);
  });
});

test.describe('Project 2 — the editor gate is offered only where it applies', () => {
  test('does not offer publish for a job that is not approved', async ({ page }) => {
    await signIn(page);
    await page.goto('/descriptions');
    // Generate a job and wait for drafts.
    await page.locator('input[placeholder*="Kestrel"], input[placeholder*="Ridge"]').first()
      .fill('E2E Gate Probe');
    await page.locator('input[placeholder*="drinkware"], input[placeholder*="home"]').first()
      .fill('drinkware');
    await page.getByRole('button', { name: /generate|submit|create/i }).first().click();

    await expect(page.locator('h3', { hasText: 'Job #' })).toBeVisible({ timeout: 45_000 });
    const publish = page.getByRole('button', { name: 'Publish to client' });
    // draft_ready shows the review actions but never the publish button.
    await expect(publish).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Approve draft' })).toBeVisible();
  });
});
