/**
 * End-to-end check of the browser-visible contract.
 *
 * This is the only test that can catch the failures the unit suites cannot:
 * that nginx actually reverse-proxies /api to the Go service, that the SPA
 * boots, and that the two product rules are enforced in the UI rather than only
 * on the server.
 *
 * Requires the full stack (`make up`). The API key is injected before the app
 * loads, because the SPA reads it from localStorage and would otherwise bounce
 * to /login.
 *
 *   API_BASE=http://localhost:8080 API_KEY=local-dev-admin-key \
 *     npx playwright test
 *
 * Point BASE_URL at the web origin (nginx), not the api: the whole point is to
 * exercise the proxy.
 */
import { defineConfig } from '@playwright/test';

const WEB = process.env.BASE_URL ?? 'http://127.0.0.1:8081';

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: WEB,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    launchOptions: {
      // The image is deliberately Linux-only; use whatever chromium the host has
      // rather than downloading one.
      executablePath: process.env.CHROME_PATH || '/usr/bin/chromium',
      args: ['--no-sandbox', '--disable-dev-shm-usage'],
    },
  },
});
