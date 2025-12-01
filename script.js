const CACHE_NAME = 'lmfrnet-offline-v24';

// モデル定義
const MODELS = {
    lmfrnet: 'models_caltech101/lmfrnet.onnx',
    lmfrnet_hires: 'models_caltech101/lmfrnet_hires.onnx',
    resnet18: 'models_caltech101/resnet18.onnx',
    mobilenetv3_large: 'models_caltech101/mobilenetv3_large.onnx'
};

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
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
    'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap'
];

let LABELS = [];
let SAMPLES = [];
let worker = null;
let currentModelName = '';
let selectedSingle = null;
let batchFiles = []; // {id, name, dataUrl}
let customSamples = []; // {id, dataUrl, label, labelName}
let uiBusy = false;

// 要素
const modelSelect = document.getElementById('model-select');
const offlinePrepBtn = document.getElementById('offline-prep-btn');
const offlineStatusEl = document.getElementById('offline-status');
const offlineChip = document.getElementById('offline-chip');
const deviceInfoEl = document.getElementById('device-info');
const standardSummaryEl = document.getElementById('standard-summary');

const heroSingleBtn = document.getElementById('hero-single-btn');
const heroBenchmarkBtn = document.getElementById('hero-benchmark-btn');

const singleDropzone = document.getElementById('single-dropzone');
const singleFileInput = document.getElementById('single-file-input');
const chooseSingleBtn = document.getElementById('choose-single-btn');
const runSingleBtn = document.getElementById('run-single-btn');
const clearSingleBtn = document.getElementById('clear-single-btn');

const batchDropzone = document.getElementById('batch-dropzone');
const batchFileInput = document.getElementById('batch-file-input');
const chooseBatchBtn = document.getElementById('choose-batch-btn');
const runBatchBtn = document.getElementById('run-batch-btn');
const batchSummaryEl = document.getElementById('batch-summary');

const runBenchmarkBtn = document.getElementById('run-benchmark-btn');

const customFileInput = document.getElementById('custom-file-input');
const customLabelSelect = document.getElementById('custom-label-select');
const addCustomBtn = document.getElementById('add-custom-btn');
const customList = document.getElementById('custom-list');
const runCustomBenchmarkBtn = document.getElementById('run-custom-benchmark-btn');

const inputCanvas = document.getElementById('input-canvas');
const predLabelEl = document.getElementById('pred-label');
const predConfEl = document.getElementById('pred-conf');
const inferTimeEl = document.getElementById('infer-time');
const singleResult = document.getElementById('single-result');

const statusPanel = document.getElementById('status-panel');
const statusText = document.getElementById('status-text');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');

const resultPanel = document.getElementById('result-panel');
const benchmarkResult = document.getElementById('benchmark-result');
const benchmarkTableBody = document.getElementById('benchmark-table-body');
const benchmarkDetailTableBody = document.getElementById('benchmark-detail-table-body');
const batchResult = document.getElementById('batch-result');
const batchResultBody = document.getElementById('batch-result-body');
const samplePreviewGrid = document.getElementById('sample-preview-grid');

// 初期化
init();

async function init() {
    deviceInfoEl.textContent = navigator.userAgent;
    initWorker();
    setupEventListeners();
    await loadLabels();
    await loadSamplesPreview();
    preloadModels();
    registerServiceWorker();
    refreshControls();
}

function initWorker() {
    worker = new Worker('worker.js');
    worker.onmessage = (e) => {
        const { type, name, error } = e.data;
        if (type === 'LOAD_DONE') {
            updateModelStatus(name, '準備完了');
        } else if (type === 'ERROR') {
            console.error('Worker Error:', error);
            alert('ワーカーでエラーが発生しました: ' + error);
        }
    };
}

function setupEventListeners() {
    heroSingleBtn.addEventListener('click', () => {
        chooseSingleBtn.click();
    });
    heroBenchmarkBtn.addEventListener('click', () => {
        runBenchmarkBtn.click();
    });

    modelSelect.addEventListener('change', () => {
        currentModelName = ''; // 切り替え時に再ロードさせる
    });

    bindDropzone(singleDropzone, (files) => {
        if (files.length > 0) prepareSinglePreview(files[0]);
    });
    bindDropzone(batchDropzone, (files) => {
        addBatchFiles(files);
    });

    chooseSingleBtn.addEventListener('click', () => singleFileInput.click());
    singleFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            prepareSinglePreview(e.target.files[0]);
        }
        singleFileInput.value = '';
    });
    clearSingleBtn.addEventListener('click', resetSinglePreview);
    runSingleBtn.addEventListener('click', runSingleInference);

    chooseBatchBtn.addEventListener('click', () => batchFileInput.click());
    batchFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) addBatchFiles(Array.from(e.target.files));
        batchFileInput.value = '';
    });
    runBatchBtn.addEventListener('click', runBatchPrediction);

    runBenchmarkBtn.addEventListener('click', () => runBenchmark(SAMPLES, '標準ベンチマーク'));
    offlinePrepBtn.addEventListener('click', prepareOfflineAssets);

    customFileInput.addEventListener('change', () => {
        addCustomBtn.disabled = !customFileInput.files.length || customLabelSelect.value === "";
    });
    customLabelSelect.addEventListener('change', () => {
        addCustomBtn.disabled = !customFileInput.files.length || customLabelSelect.value === "";
    });
    addCustomBtn.addEventListener('click', addCustomSample);
    runCustomBenchmarkBtn.addEventListener('click', () => runBenchmark(customSamples, 'カスタム（ラベル付き）'));
}

function bindDropzone(zone, handler) {
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files || []);
        if (files.length > 0) handler(files);
    });
}

async function loadLabels() {
    try {
        const res = await fetch(`labels.json?t=${Date.now()}`);
        LABELS = await res.json();
        populateCustomLabelSelect();
    } catch (e) {
        console.error('ラベルの読み込みに失敗', e);
        alert('labels.json を読み込めませんでした。');
    }
}

async function loadSamplesPreview() {
    try {
        const res = await fetch(`samples.json?t=${Date.now()}`);
        SAMPLES = await res.json();
        standardSummaryEl.textContent = `${SAMPLES.length}枚 / Caltech-101`;
        renderSamplePreview();
    } catch (e) {
        console.error('samples.json の読み込みに失敗', e);
        standardSummaryEl.textContent = '読み込み失敗';
    }
}

function renderSamplePreview() {
    if (!samplePreviewGrid) return;
    samplePreviewGrid.innerHTML = '';
    const subset = SAMPLES.slice(0, 8);
    subset.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.innerHTML = `
            <img src="samples/${item.filename}" alt="${item.category}" loading="lazy">
            <p class="tiny" style="margin:6px 0 2px;">${item.category}</p>
            <p class="tiny muted" style="margin:0;">${item.filename}</p>
            <p class="tiny muted" style="margin:2px 0 0;">Label ID: ${item.label}</p>
        `;
        samplePreviewGrid.appendChild(card);
    });
}

function populateCustomLabelSelect() {
    customLabelSelect.innerHTML = '<option value="">正解ラベルを選択...</option>';
    LABELS.forEach((label, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${label} (ID: ${index})`;
        customLabelSelect.appendChild(option);
    });
    customLabelSelect.disabled = false;
}

function prepareSinglePreview(file) {
    if (!file.type.startsWith('image/')) {
        alert('画像ファイルを選択してください。');
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            const displayWidth = Math.min(480, img.width);
            const scale = displayWidth / img.width;
            inputCanvas.width = displayWidth;
            inputCanvas.height = img.height * scale;
            const ctx = inputCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, inputCanvas.width, inputCanvas.height);
            selectedSingle = { file, dataUrl: e.target.result };
            singleResult.classList.remove('hidden');
            refreshControls();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function resetSinglePreview() {
    const ctx = inputCanvas.getContext('2d');
    ctx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
    selectedSingle = null;
    predLabelEl.textContent = '---';
    predConfEl.textContent = '---%';
    inferTimeEl.textContent = '--- ms';
    singleResult.classList.add('hidden');
    refreshControls();
}

async function runSingleInference() {
    if (!selectedSingle) return;
    setBusy(true);
    showStatus('推論を実行中...');
    await ensureModelLoaded();

    try {
        const offCanvas = document.createElement('canvas');
        offCanvas.width = 224; offCanvas.height = 224;
        const offCtx = offCanvas.getContext('2d');
        offCtx.drawImage(inputCanvas, 0, 0, 224, 224);
        const imageData = offCtx.getImageData(0, 0, 224, 224);
        const { probs, time } = await runInferenceOnWorker(imageData.data);
        const { idx, prob } = topProbability(probs);

        predLabelEl.textContent = LABELS[idx] || `クラス ${idx}`;
        predConfEl.textContent = `${(prob * 100).toFixed(1)}%`;
        inferTimeEl.textContent = `${time.toFixed(2)} ms`;

        hideStatus();
    } catch (e) {
        console.error('推論エラー', e);
        alert(`推論中にエラーが発生しました: ${e.message || e}`);
        hideStatus();
    } finally {
        setBusy(false);
    }
}

async function runBenchmark(dataset, labelName = 'ベンチマーク') {
    if (!dataset || dataset.length === 0) {
        alert('評価用の画像がありません。');
        return;
    }
    if (uiBusy) return;

    setBusy(true);
    benchmarkResult.classList.add('hidden');
    batchResult.classList.add('hidden');
    benchmarkTableBody.innerHTML = '';
    benchmarkDetailTableBody.innerHTML = '';
    showStatus(`${labelName}を実行中...`, true);
    updateProgress(0, `0 / ${dataset.length}`);

    try {
        await ensureModelLoaded();
        const dummy = new Uint8ClampedArray(224 * 224 * 4);
        await runInferenceOnWorker(dummy); // ウォームアップ

        let correct = 0;
        let totalTime = 0;
        let totalConf = 0;
        const details = [];

        for (let i = 0; i < dataset.length; i++) {
            const sample = dataset[i];
            const img = await loadImage(sample);
            const imageData = get224ImageData(img);
            const { probs, time } = await runInferenceOnWorker(imageData.data);
            const { idx, prob } = topProbability(probs);

            const isCorrect = idx === sample.label;
            if (isCorrect) {
                correct++;
                totalConf += prob;
            }
            totalTime += time;

            details.push({
                index: i + 1,
                imageSrc: sample.dataUrl || `samples/${sample.filename}`,
                correctLabel: LABELS[sample.label] || 'N/A',
                predLabel: LABELS[idx] || `クラス ${idx}`,
                confidence: prob,
                isCorrect,
                filename: sample.filename || `Custom ${i + 1}`
            });

            const progress = Math.round(((i + 1) / dataset.length) * 100);
            updateProgress(progress, `${i + 1} / ${dataset.length}`);
            if (i % 5 === 0) await new Promise((r) => setTimeout(r, 0));
        }

        const acc = (correct / dataset.length) * 100;
        const avgTime = totalTime / dataset.length;
        const avgConf = correct > 0 ? (totalConf / correct) * 100 : 0;

        const metrics = [
            { name: 'モデル', value: modelSelect.value },
            { name: 'データセット', value: `${labelName} (${dataset.length}枚)` },
            { name: '正解率', value: `${acc.toFixed(2)}%` },
            { name: '平均推論時間', value: `${avgTime.toFixed(2)} ms` },
            { name: '平均確信度 (正解時)', value: `${avgConf.toFixed(2)}%` }
        ];

        benchmarkTableBody.innerHTML = metrics.map((m) => `
            <tr>
                <td>${m.name}</td>
                <td>${m.value}</td>
            </tr>
        `).join('');

        benchmarkDetailTableBody.innerHTML = details.map((d) => `
            <tr class="${d.isCorrect ? 'correct-row' : 'incorrect-row'}">
                <td>${d.index}</td>
                <td><img src="${d.imageSrc}" style="width: 50px; height: 50px; object-fit: cover;" loading="lazy"></td>
                <td>${d.correctLabel}</td>
                <td>${d.predLabel}</td>
                <td>${(d.confidence * 100).toFixed(1)}%</td>
                <td>${d.isCorrect ? '○' : '×'}</td>
            </tr>
        `).join('');

        resultPanel.classList.remove('hidden');
        benchmarkResult.classList.remove('hidden');
        hideStatus();
    } catch (e) {
        console.error('ベンチマークエラー', e);
        alert(`ベンチマーク中にエラーが発生しました: ${e.message || e}`);
        hideStatus();
    } finally {
        setBusy(false);
    }
}

async function runBatchPrediction() {
    if (batchFiles.length === 0 || uiBusy) return;
    setBusy(true);
    batchResultBody.innerHTML = '';
    benchmarkResult.classList.add('hidden');
    batchResult.classList.add('hidden');
    showStatus('ローカル画像を推論中...', true);
    updateProgress(0, `0 / ${batchFiles.length}`);

    try {
        await ensureModelLoaded();
        const dummy = new Uint8ClampedArray(224 * 224 * 4);
        await runInferenceOnWorker(dummy); // ウォームアップ

        const rows = [];
        for (let i = 0; i < batchFiles.length; i++) {
            const item = batchFiles[i];
            const img = await loadImage({ dataUrl: item.dataUrl });
            const imageData = get224ImageData(img);
            const { probs, time } = await runInferenceOnWorker(imageData.data);
            const { idx, prob } = topProbability(probs);

            rows.push(`
                <tr>
                    <td>${i + 1}</td>
                    <td><img src="${item.dataUrl}" style="width:50px;height:50px;object-fit:cover;" loading="lazy"></td>
                    <td>${LABELS[idx] || `クラス ${idx}`}</td>
                    <td>${(prob * 100).toFixed(1)}%</td>
                    <td>${time.toFixed(2)} ms</td>
                </tr>
            `);

            const progress = Math.round(((i + 1) / batchFiles.length) * 100);
            updateProgress(progress, `${i + 1} / ${batchFiles.length}`);
            if (i % 5 === 0) await new Promise((r) => setTimeout(r, 0));
        }

        batchResultBody.innerHTML = rows.join('');
        resultPanel.classList.remove('hidden');
        batchResult.classList.remove('hidden');
        hideStatus();
    } catch (e) {
        console.error('バッチ推論エラー', e);
        alert(`推論中にエラーが発生しました: ${e.message || e}`);
        hideStatus();
    } finally {
        setBusy(false);
    }
}

async function prepareOfflineAssets() {
    if (!('caches' in window)) {
        alert('このブラウザはキャッシュ API に対応していません。');
        return;
    }
    if (uiBusy) return;

    setBusy(true);
    showStatus('オフライン準備中...', true);
    offlineStatusEl.textContent = 'ダウンロード中...';

    try {
        const cache = await caches.open(CACHE_NAME);
        const sampleAssets = SAMPLES.map((s) => `samples/${s.filename}`);
        const assets = Array.from(new Set([...CORE_ASSETS, ...sampleAssets, ...Object.values(MODELS)]));

        let done = 0;
        for (const url of assets) {
            try {
                const res = await fetch(url);
                await cache.put(url, res.clone());
            } catch (e) {
                console.warn('キャッシュに失敗:', url, e);
            }
            done++;
            const percent = Math.round((done / assets.length) * 100);
            updateProgress(percent, `${done} / ${assets.length}`);
        }

        offlineStatusEl.textContent = '端末にキャッシュ完了。オフラインでも実行できます。';
        offlineChip.textContent = 'オフライン準備: 完了';
        offlineChip.classList.add('good');
        hideStatus();
    } catch (e) {
        console.error('オフライン準備エラー', e);
        offlineStatusEl.textContent = 'キャッシュに失敗しました。';
        alert(`キャッシュ中にエラー: ${e.message || e}`);
        hideStatus();
    } finally {
        setBusy(false);
    }
}

function updateModelStatus(name, status) {
    const container = document.getElementById('model-status-container');
    const statusId = `status-${name}`;
    let el = document.getElementById(statusId);
    if (!el) {
        el = document.createElement('div');
        el.id = statusId;
        el.className = 'status-line';
        container.appendChild(el);
    }
    const option = document.querySelector(`#model-select option[value="${name}"]`);
    const modelDisplayName = option ? option.textContent : name;
    el.textContent = `${modelDisplayName}: ${status}`;

    if (status.includes('完了')) el.style.color = '#37e8c0';
    else if (status.includes('エラー')) el.style.color = '#ff7b7b';
    else el.style.color = '#ffb347';
}

async function preloadModels() {
    const loadOrder = ['lmfrnet', 'lmfrnet_hires', 'resnet18', 'mobilenetv3_large'];
    for (const name of loadOrder) {
        if (MODELS[name]) {
            try {
                updateModelStatus(name, '読み込み中...');
                await loadModelOnWorker(name, MODELS[name]);
                updateModelStatus(name, '準備完了');
            } catch (e) {
                console.warn(`Error preloading ${name}`, e);
                updateModelStatus(name, 'エラー');
            }
        }
    }
}

function topProbability(probs) {
    let maxProb = -1;
    let idx = -1;
    for (let i = 0; i < probs.length; i++) {
        if (probs[i] > maxProb) {
            maxProb = probs[i];
            idx = i;
        }
    }
    return { idx, prob: maxProb };
}

function get224ImageData(img) {
    const offCanvas = document.createElement('canvas');
    offCanvas.width = 224; offCanvas.height = 224;
    const ctx = offCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 224, 224);
    return ctx.getImageData(0, 0, 224, 224);
}

async function loadImage(sample) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = sample.dataUrl || `samples/${sample.filename}`;
    });
}

function showStatus(message, showProgress = false) {
    statusPanel.classList.remove('hidden');
    statusText.textContent = message;
    if (showProgress) {
        progressContainer.classList.remove('hidden');
    } else {
        progressContainer.classList.add('hidden');
    }
}

function hideStatus() {
    statusPanel.classList.add('hidden');
    progressBar.style.width = '0%';
}

function updateProgress(percent, text) {
    progressBar.style.width = `${percent}%`;
    progressLabel.textContent = text;
}

function setBusy(state) {
    uiBusy = state;
    refreshControls();
}

function refreshControls() {
    const disabled = uiBusy;
    runSingleBtn.disabled = disabled || !selectedSingle;
    runBenchmarkBtn.disabled = disabled;
    runBatchBtn.disabled = disabled || batchFiles.length === 0;
    modelSelect.disabled = disabled;
    offlinePrepBtn.disabled = disabled;
    addCustomBtn.disabled = disabled || !customFileInput.files.length || customLabelSelect.value === "";
    runCustomBenchmarkBtn.disabled = disabled || customSamples.length === 0;
}

async function ensureModelLoaded() {
    const modelName = modelSelect.value;
    if (currentModelName === modelName) return;
    updateModelStatus(modelName, '切り替え中...');
    await loadModelOnWorker(modelName, MODELS[modelName]);
    currentModelName = modelName;
}

function runInferenceOnWorker(pixelData) {
    return new Promise((resolve, reject) => {
        const handler = (e) => {
            const { type, result, error } = e.data;
            if (type === 'RUN_DONE') {
                worker.removeEventListener('message', handler);
                resolve(result);
            } else if (type === 'ERROR') {
                worker.removeEventListener('message', handler);
                reject(error);
            }
        };
        worker.addEventListener('message', handler);
        worker.postMessage({ type: 'RUN', data: { pixelData } });
    });
}

function loadModelOnWorker(name, url) {
    return new Promise((resolve, reject) => {
        const handler = (e) => {
            const { type, name: loadedName, error } = e.data;
            if (type === 'LOAD_DONE' && loadedName === name) {
                worker.removeEventListener('message', handler);
                resolve();
            } else if (type === 'ERROR') {
                worker.removeEventListener('message', handler);
                reject(error);
            }
        };
        worker.addEventListener('message', handler);
        worker.postMessage({ type: 'LOAD', data: { name, url } });
    });
}

function addBatchFiles(files) {
    const imageFiles = files.filter((f) => f.type.startsWith('image/'));
    const readers = imageFiles.map(async (file) => {
        const dataUrl = await readFileAsDataURL(file);
        batchFiles.push({
            id: Date.now() + Math.random(),
            name: file.name,
            dataUrl
        });
    });
    Promise.all(readers).then(() => {
        batchSummaryEl.textContent = `${batchFiles.length}枚選択中`;
        runBatchBtn.disabled = batchFiles.length === 0 || uiBusy;
    });
}

async function addCustomSample() {
    const file = customFileInput.files[0];
    const labelIdx = parseInt(customLabelSelect.value, 10);
    if (!file || Number.isNaN(labelIdx)) return;
    const dataUrl = await readFileAsDataURL(file);
    customSamples.push({
        id: Date.now() + Math.random(),
        dataUrl,
        label: labelIdx,
        labelName: LABELS[labelIdx]
    });
    renderCustomList();
    customFileInput.value = '';
    customLabelSelect.value = '';
    addCustomBtn.disabled = true;
    runCustomBenchmarkBtn.classList.remove('hidden');
}

function renderCustomList() {
    customList.innerHTML = '';
    if (customSamples.length === 0) {
        customList.classList.add('hidden');
        return;
    }
    customList.classList.remove('hidden');
    customSamples.forEach((sample) => {
        const div = document.createElement('div');
        div.className = 'custom-item';
        div.innerHTML = `
            <img src="${sample.dataUrl}">
            <div class="label-tag">${sample.labelName}</div>
            <button class="delete-btn" onclick="deleteCustomSample(${sample.id})">×</button>
        `;
        customList.appendChild(div);
    });
    refreshControls();
}

window.deleteCustomSample = function (id) {
    customSamples = customSamples.filter((s) => s.id !== id);
    renderCustomList();
    if (customSamples.length === 0) {
        runCustomBenchmarkBtn.classList.add('hidden');
    }
    refreshControls();
};

function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function registerServiceWorker() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('sw.js').catch((err) => {
            console.warn('ServiceWorker 登録失敗:', err);
        });
    }
}
