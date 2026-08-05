const CACHE = 'deepweather-v3';
const ASSETS = ['./','./index.html','./styles.css','./app.js','./manifest.json',
                './scaler.json','./lstm_model.onnx','./icon-192.png','./icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks =>
    Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ).then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // live weather is always fetched fresh, never served stale from cache
  if (url.hostname.endsWith('open-meteo.com')) return;
  // ignoreSearch so cache-busting query strings (app.js?v=5) still match offline
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then(hit => hit || fetch(e.request))
  );
});
