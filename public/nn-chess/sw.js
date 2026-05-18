// nn-chess service worker.
//
// Strategy:
//   - Navigations under /nn-chess/ are network-first with a cached
//     shell fallback so the app still loads offline.
//   - Other GETs in scope are cache-first with a background fetch so
//     hashed Astro assets get cached on first hit.
//
// Bump CACHE_VERSION to force clients off old caches after a deploy
// that changes the SW itself or the shell layout.

const CACHE_VERSION = 'v1';
const CACHE_NAME = `nn-chess-${CACHE_VERSION}`;
const SCOPE_PATH = '/nn-chess/';
const SHELL = [
  '/nn-chess/',
  '/nn-chess/sparring/',
  '/nn-chess/manifest.webmanifest',
  '/nn-chess/icon-192.png',
  '/nn-chess/icon-512.png',
  '/nn-chess/icon-512-maskable.png',
  '/nn-chess/apple-touch-icon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      // Don't fail install if one of the shell URLs is missing in this
      // build — just cache what we can.
      Promise.all(SHELL.map((u) => cache.add(u).catch(() => null))),
    ),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))),
    ),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith(SCOPE_PATH)) return;

  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE_NAME).then((c) => c.put(req, copy));
          return res;
        })
        .catch(() =>
          caches.match(req).then((hit) => hit || caches.match('/nn-chess/')),
        ),
    );
    return;
  }

  event.respondWith(
    caches.match(req).then((hit) => {
      if (hit) return hit;
      return fetch(req).then((res) => {
        if (res.ok && res.type === 'basic') {
          const copy = res.clone();
          caches.open(CACHE_NAME).then((c) => c.put(req, copy));
        }
        return res;
      });
    }),
  );
});
