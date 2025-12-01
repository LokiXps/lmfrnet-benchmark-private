const CACHE_NAME = 'lmfrnet-benchmark-v26';
const CORE_ASSETS = [
    './',
    './index.html',
    './style.css',
    './script.js',
    './worker.js',
    './labels.json',
    './samples.json',
    './gallery.html',
    './manifest.json',
    './models_caltech101/lmfrnet.onnx',
    './models_caltech101/lmfrnet_hires.onnx',
    './models_caltech101/resnet18.onnx',
    './models_caltech101/mobilenetv3_large.onnx',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
    'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(CORE_ASSETS)).catch((err) => {
            console.warn('SW install cache error', err);
        })
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((names) => Promise.all(
            names.map((name) => (name !== CACHE_NAME ? caches.delete(name) : Promise.resolve()))
        ))
    );
});

self.addEventListener('fetch', (event) => {
    if (event.request.method !== 'GET') return;
    event.respondWith(
        caches.open(CACHE_NAME).then((cache) => cache.match(event.request).then((cached) => {
            const fetchAndCache = fetch(event.request).then((resp) => {
                const shouldCache =
                    event.request.url.startsWith(self.location.origin) ||
                    event.request.url.includes('cdn.jsdelivr.net') ||
                    event.request.url.includes('fonts.gstatic.com') ||
                    event.request.url.includes('fonts.googleapis.com');

                if (shouldCache && resp && resp.status === 200) {
                    cache.put(event.request, resp.clone()).catch(() => { });
                }
                return resp;
            });

            return cached || fetchAndCache;
        }))
    );
});
