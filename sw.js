const CACHE_NAME = 'lmfrnet-benchmark-v33';
const ASSETS = [
    './',
    './index.html',
    './style.css',
    './manifest.json',
    './labels.json',
    './samples.json',
    './icon-192.png',
    './icon-512.png',
    './worker.js',
    './models_caltech101/lmfrnet.onnx',
    './models_caltech101/lmfrnet_hires.onnx',
    './models_caltech101/mobilenetv3_large.onnx',
    './models_caltech101/resnet18.onnx',
    './ort.min.js',
    './ort-wasm.wasm',
    './ort-wasm-simd.wasm',
    './ort-wasm-threaded.wasm',
    './ort-wasm-simd-threaded.wasm',
    './ort-wasm-simd-threaded.mjs',
    './ort-wasm-simd-threaded.jsep.mjs',
    './ort-wasm-threaded.jsep.mjs',
    './tailwind.min.js'
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
                // Only fallback to index.html for navigation requests (page loads)
                if (event.request.mode === 'navigate') {
                    return caches.match('./index.html');
                }
            });
        })
    );
});
