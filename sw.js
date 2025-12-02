const CACHE_NAME = 'lmfrnet-benchmark-v37';
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
    // CDNキャッシュ用の別枠 (リソースの実体はCDNから取る)
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.jsep.mjs',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.jsep.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.jsep.js',
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
