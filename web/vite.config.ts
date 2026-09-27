import { defineConfig } from 'vitest/config';
import solid from 'vite-plugin-solid';

export default defineConfig({
  plugins: [solid()],
  build: {
    target: 'es2020',
    outDir: 'dist',
    sourcemap: true,
    // Must NOT be "assets". ASSET_PUBLIC_BASE_URL=/assets is the product-photo
    // route, and nginx serves it with `location /assets/ { alias /data/assets/ }`.
    // Vite's default output dir is also "assets", so the app's own bundle was
    // shadowed by that alias: index.html referenced /assets/index-<hash>.js,
    // nginx looked in the photo volume, returned 404, and the SPA rendered a
    // blank page. Keep the two URL spaces disjoint.
    assetsDir: 'static',
  },
  server: {
    port: 5173,
    // `npm run dev` with an empty VITE_API_BASE calls /api/v1 same-origin, so
    // the dev server needs the same reverse proxy nginx has in the image.
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE || 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
    globals: false,
  },
});
