const CACHE_NAME = 'lmfrnet-benchmark-v30';
const ASSETS = [
    './',
    './index.html',
    './manifest.json',
    './labels.json',
    './samples.json',
    './icon-192.png',
    './icon-512.png',
    './models_caltech101/lmfrnet.onnx',
    './models_caltech101/lmfrnet_hires.onnx',
    './models_caltech101/mobilenetv3_large.onnx',
    './models_caltech101/resnet18.onnx',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
    'https://cdn.tailwindcss.com'
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
            }).catch(() => caches.match('./index.html'));
        })
    );
});
