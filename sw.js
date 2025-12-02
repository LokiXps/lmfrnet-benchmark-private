const CACHE_NAME = 'lmfrnet-benchmark-v1';
const ASSETS = [
    './',
    './index.html',
    './style.css',
    './manifest.json',
    './labels.json',
    './samples.json',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm',
    'https://cdn.tailwindcss.com'
];

// Install Event
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(ASSETS))
    );
});

// Fetch Event
self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // Cache hit - return response
                if (response) {
                    return response;
                }
                // Clone the request
                const fetchRequest = event.request.clone();

                return fetch(fetchRequest).then(
                    (response) => {
                        // Check if we received a valid response
                        if (!response || response.status !== 200 || response.type !== 'basic') {
                            return response;
                        }

                        // Clone the response
                        const responseToCache = response.clone();

                        caches.open(CACHE_NAME)
                            .then((cache) => {
                                // Cache new requests (including models and images)
                                cache.put(event.request, responseToCache);
                            });

                        return response;
                    }
                );
            })
    );
});

// Activate Event (Cleanup old caches)
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
});
