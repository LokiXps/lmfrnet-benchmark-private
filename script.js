// CIFAR-10 Labels
const LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
];

// Elements
const fileInput = document.getElementById('file-input');
const dropArea = document.getElementById('drop-area');
const modelSelect = document.getElementById('model-select');
const runBtn = document.getElementById('run-btn');
const inputCanvas = document.getElementById('input-canvas');
const ctx = inputCanvas.getContext('2d');
const resultPanel = document.getElementById('result-panel');
const predLabelEl = document.getElementById('pred-label');
const predConfEl = document.getElementById('pred-conf');
const inferTimeEl = document.getElementById('infer-time');
const deviceInfoEl = document.getElementById('device-info');
const spinner = document.querySelector('.loading-spinner');
const btnText = document.querySelector('.btn-text');

let currentSession = null;
let currentModelName = '';
let imageLoaded = false;

// Initialize
async function init() {
    // Check WebGPU support
    const userAgent = navigator.userAgent;
    deviceInfoEl.textContent = userAgent;

    // Setup Event Listeners
    dropArea.addEventListener('click', () => fileInput.click());

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('dragover');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    runBtn.addEventListener('click', runInference);

    // Pre-load default model? Maybe not, let user choose.
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            // Draw to canvas (display)
            // We keep the canvas size reasonable for display, but will resize for inference
            const displayWidth = Math.min(300, img.width);
            const scale = displayWidth / img.width;
            inputCanvas.width = displayWidth;
            inputCanvas.height = img.height * scale;
            ctx.drawImage(img, 0, 0, inputCanvas.width, inputCanvas.height);

            imageLoaded = true;
            updateButtonState();
            resultPanel.classList.add('hidden');
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function updateButtonState() {
    runBtn.disabled = !imageLoaded;
}

async function loadModel(modelName) {
    if (currentSession && currentModelName === modelName) {
        return currentSession;
    }

    btnText.textContent = `Loading ${modelName}...`;
    spinner.classList.remove('hidden');
    runBtn.disabled = true;

    try {
        // Path to ONNX file
        const modelPath = `../models_onnx/${modelName}.onnx`;

        // Create session
        // Try WebGPU first, then WASM
        const options = {
            executionProviders: ['wasm'], // Start with WASM for stability, or 'webgpu' if available
            // executionProviders: ['webgpu', 'wasm'], 
        };

        currentSession = await ort.InferenceSession.create(modelPath, options);
        currentModelName = modelName;
        console.log(`Model ${modelName} loaded.`);
        return currentSession;

    } catch (e) {
        console.error("Failed to load model", e);
        alert(`Failed to load model: ${e.message}`);
        return null;
    } finally {
        spinner.classList.add('hidden');
        btnText.textContent = "Run Inference";
        runBtn.disabled = false;
    }
}

async function preprocess(ctx, width, height) {
    // Resize to 32x32 (CIFAR-10 size)
    const offScreenCanvas = document.createElement('canvas');
    offScreenCanvas.width = 32;
    offScreenCanvas.height = 32;
    const offCtx = offScreenCanvas.getContext('2d');

    // Draw original canvas content to 32x32
    offCtx.drawImage(inputCanvas, 0, 0, 32, 32);

    const imageData = offCtx.getImageData(0, 0, 32, 32);
    const { data } = imageData;

    // Convert to Float32 Tensor [1, 3, 32, 32]
    // Normalized: (x - mean) / std
    // CIFAR-10 mean/std
    const mean = [0.4914, 0.4822, 0.4465];
    const std = [0.2023, 0.1994, 0.2010];

    const float32Data = new Float32Array(1 * 3 * 32 * 32);

    for (let i = 0; i < 32 * 32; i++) {
        const r = data[i * 4] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;

        // R channel
        float32Data[i] = (r - mean[0]) / std[0];
        // G channel
        float32Data[32 * 32 + i] = (g - mean[1]) / std[1];
        // B channel
        float32Data[2 * 32 * 32 + i] = (b - mean[2]) / std[2];
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 32, 32]);
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

async function runInference() {
    if (!imageLoaded) return;

    const modelName = modelSelect.value;
    const session = await loadModel(modelName);
    if (!session) return;

    btnText.textContent = "Running...";
    spinner.classList.remove('hidden');
    runBtn.disabled = true;

    // Allow UI to update
    await new Promise(r => setTimeout(r, 10));

    try {
        const inputTensor = await preprocess(ctx, inputCanvas.width, inputCanvas.height);

        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;

        const start = performance.now();
        const results = await session.run(feeds);
        const end = performance.now();

        const output = results[session.outputNames[0]].data;

        // Handle case where model output size doesn't match labels (e.g. LMFRNet trained with 100 classes)
        let probs;
        if (output.length > LABELS.length) {
            // Slice the first 10 logits
            const logits = Array.from(output).slice(0, LABELS.length);
            probs = softmax(logits);
        } else {
            probs = softmax(Array.from(output));
        }

        // Find max
        let maxProb = -1;
        let maxIndex = -1;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }

        // Display results
        predLabelEl.textContent = LABELS[maxIndex];
        predConfEl.textContent = `${(maxProb * 100).toFixed(1)}%`;
        inferTimeEl.textContent = (end - start).toFixed(2);

        resultPanel.classList.remove('hidden');

    } catch (e) {
        console.error("Inference failed", e);
        alert(`Inference failed: ${e.message}`);
    } finally {
        spinner.classList.add('hidden');
        btnText.textContent = "Run Inference";
        runBtn.disabled = false;
    }
}

init();
