importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// Configure ONNX Runtime for Mobile/Web
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let session = null;
let currentModelName = '';

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        if (type === 'LOAD') {
            await loadModel(data.name, data.url);
            self.postMessage({ type: 'LOAD_DONE', name: data.name });
        } else if (type === 'RUN') {
            const result = await runInference(data.pixelData);
            self.postMessage({ type: 'RUN_DONE', result: result });
        }
    } catch (err) {
        self.postMessage({ type: 'ERROR', error: err.message });
    }
};

async function loadModel(name, url) {
    if (currentModelName === name && session) return;

    try {
        // Fetch model buffer to ensure clean loading
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();

        const options = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'basic'
        };
        session = await ort.InferenceSession.create(buffer, options);
        currentModelName = name;
    } catch (e) {
        throw e;
    }
}

async function runInference(pixelData) {
    if (!session) throw new Error("Session not loaded");

    // Preprocess: RGBA -> RGB -> Normalize -> CHW
    // pixelData is Uint8ClampedArray (RGBA) from canvas
    const width = 224;
    const height = 224;
    const float32Data = new Float32Array(1 * 3 * width * height);

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0; i < width * height; i++) {
        // Normalize 0-255 to 0.0-1.0
        const r = pixelData[i * 4] / 255.0;
        const g = pixelData[i * 4 + 1] / 255.0;
        const b = pixelData[i * 4 + 2] / 255.0;

        // Standardize
        float32Data[i] = (r - mean[0]) / std[0];
        float32Data[width * height + i] = (g - mean[1]) / std[1];
        float32Data[2 * width * height + i] = (b - mean[2]) / std[2];
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, width, height]);

    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    const start = performance.now();
    const results = await session.run(feeds);
    const end = performance.now();

    const output = results[session.outputNames[0]].data;

    // Softmax
    const arr = Array.from(output);
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(x => x / sum);

    return { probs, time: end - start };
}
