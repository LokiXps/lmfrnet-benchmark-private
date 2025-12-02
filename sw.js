const CACHE_NAME = 'lmfrnet-web-v40';
const ASSETS = [
    './',
    './index.html',
    './style.css',
    './manifest.json',
    './labels.json',
    './samples.json',
    './icon-192.png',
    './icon-512.png',
    './main.js',
    './models_caltech101/lmfrnet.onnx',
    './models_caltech101/lmfrnet_hires.onnx',
    './models_caltech101/mobilenetv3_large.onnx',
    './models_caltech101/resnet18.onnx',
    // CDNキャッシュ（初回オンライン時に取っておく）
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.wasm.min.js'
];

self.addEventListener('install', (event) => {
    self.skipWaiting();
    event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS)));
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((names) => Promise.all(
            names.map((name) => (name !== CACHE_NAME ? caches.delete(name) : Promise.resolve()))
        )).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (event) => {
    if (event.request.method !== 'GET') return;
    event.respondWith(
        caches.match(event.request, { ignoreVary: true, ignoreSearch: true }).then((cached) => {
            if (cached) return cached;
            return fetch(event.request).then((resp) => {
                if (!resp || resp.status !== 200) return resp;
                caches.open(CACHE_NAME).then((cache) => cache.put(event.request, resp.clone())).catch(() => { });
                return resp;
            }).catch(() => {
                if (event.request.mode === 'navigate') {
                    return caches.match('./index.html');
                }
            });
        })
    );
});
