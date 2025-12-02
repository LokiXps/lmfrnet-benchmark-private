// モデル一覧
const MODEL_OPTIONS = [
  { name: "LMFRNet (Default)", path: "./models_caltech101/lmfrnet.onnx" },
  { name: "LMFRNet Hires", path: "./models_caltech101/lmfrnet_hires.onnx" },
  { name: "MobileNetV3 Large", path: "./models_caltech101/mobilenetv3_large.onnx" },
  { name: "ResNet18", path: "./models_caltech101/resnet18.onnx" }
];

let LABELS = [];
let BENCHMARK_DATA = [];
let session = null;
let currentModelName = "";
let isRunning = false;

// DOM
const statusEl = document.getElementById('model-status');
const modelSelect = document.getElementById('model-select');
const benchTableBody = document.getElementById('bench-table-body');
const benchAccEl = document.getElementById('bench-acc');
const benchTimeEl = document.getElementById('bench-time');
const benchConfEl = document.getElementById('bench-conf');
const benchProgEl = document.getElementById('bench-prog');
const benchProgressBar = document.getElementById('bench-progress-bar');
const benchProgressContainer = document.getElementById('bench-progress-container');
const datasetGrid = document.getElementById('dataset-grid');
const btnRunBench = document.getElementById('btn-run-bench');
const btnRunCustom = document.getElementById('btn-run-custom');
const customResultPanel = document.getElementById('custom-result-panel');
const modelNameDisplays = document.querySelectorAll('.model-name-display');

let customImages = [];

// 初期化
window.addEventListener('load', async () => {
  try {
    // Service Worker
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('./sw.js').catch(() => { });
    }

    // ONNX Runtime Web: CDN を利用
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
    ort.env.wasm.numThreads = 1; // iPhone安定優先
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;

    initModelSelect();
    await loadLabels();
    await loadSamples();
    renderDatasetGrid();
    await loadModel(MODEL_OPTIONS[0].path);
  } catch (e) {
    console.error("Init Error:", e);
    alert("初期化エラー: " + e.message);
  }
});

function initModelSelect() {
  modelSelect.innerHTML = '';
  MODEL_OPTIONS.forEach(opt => {
    const option = document.createElement('option');
    option.value = opt.path;
    option.textContent = opt.name;
    modelSelect.appendChild(option);
  });
  modelSelect.value = MODEL_OPTIONS[0].path;
  modelSelect.addEventListener('change', (e) => {
    loadModel(e.target.value);
  });
}

async function loadLabels() {
  const res = await fetch('labels.json');
  LABELS = await res.json();
}

async function loadSamples() {
  const res = await fetch('samples.json');
  BENCHMARK_DATA = await res.json();
}

async function loadModel(path) {
  const name = path.split('/').pop().replace('.onnx', '');
  currentModelName = name;
  updateModelNameDisplay(name);

  statusEl.textContent = "ロード中...";
  statusEl.className = "text-xs px-2 py-1 rounded bg-yellow-200 text-yellow-800 font-bold";
  enableRunButtons(false);

  try {
    session = await ort.InferenceSession.create(path, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all'
    });
    statusEl.textContent = "準備完了";
    statusEl.className = "text-xs px-2 py-1 rounded bg-green-200 text-green-800 font-bold";
    enableRunButtons(true);
  } catch (e) {
    console.error("Model load failed:", e);
    statusEl.textContent = "エラー";
    statusEl.className = "text-xs px-2 py-1 rounded bg-red-200 text-red-800 font-bold";
    alert("モデル読み込みに失敗しました: " + e.message);
  }
}

function updateModelNameDisplay(name) {
  modelNameDisplays.forEach(el => el.textContent = `Model: ${name}`);
}

function enableRunButtons(enabled) {
  btnRunBench.disabled = !enabled;
  btnRunCustom.disabled = !enabled || customImages.length === 0;
}

async function runInference(blob) {
  // 1. 前処理（224センタークロップ、CHW、正規化）
  const pixelData = await loadAndCenterCrop(blob, 224);
  const size = 224;
  const float32Data = new Float32Array(3 * size * size);
  for (let i = 0; i < size * size; i++) {
    const r = pixelData[i * 4] / 255.0;
    const g = pixelData[i * 4 + 1] / 255.0;
    const b = pixelData[i * 4 + 2] / 255.0;
    float32Data[i] = (r - 0.485) / 0.229;
    float32Data[size * size + i] = (g - 0.456) / 0.224;
    float32Data[2 * size * size + i] = (b - 0.406) / 0.225;
  }

  const tensor = new ort.Tensor('float32', float32Data, [1, 3, size, size]);
  const feeds = { [session.inputNames[0]]: tensor };

  const start = performance.now();
  const results = await session.run(feeds);
  const end = performance.now();

  const output = results[session.outputNames[0]].data;
  return { probs: softmax(Array.from(output)), time: end - start };
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function argmax(probs) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }
  return { bestIndex: best, bestProb: probs[best] };
}

// 224センタークロップ
async function loadAndCenterCrop(blob, target) {
  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = url;
    });

    const canvas = document.createElement('canvas');
    canvas.width = target;
    canvas.height = target;
    const ctx = canvas.getContext('2d');

    const scale = Math.max(target / img.width, target / img.height);
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    const dx = (target - scaledWidth) / 2;
    const dy = (target - scaledHeight) / 2;

    ctx.drawImage(img, dx, dy, scaledWidth, scaledHeight);
    return ctx.getImageData(0, 0, target, target).data;
  } finally {
    URL.revokeObjectURL(url);
  }
}

// ベンチマーク
async function runBenchmark() {
  if (isRunning) return;
  if (!session || BENCHMARK_DATA.length === 0) {
    alert("モデルまたはデータが準備できていません。");
    return;
  }
  isRunning = true;
  enableRunButtons(false);

  benchTableBody.innerHTML = '';
  benchAccEl.textContent = '--';
  benchTimeEl.textContent = '--';
  benchConfEl.textContent = '--';
  benchProgEl.textContent = '0/200';
  benchProgressBar.style.width = '0%';
  benchProgressContainer.style.display = 'block';

  let correct = 0;
  let totalTime = 0;
  let totalConf = 0;

  for (let i = 0; i < BENCHMARK_DATA.length; i++) {
    const item = BENCHMARK_DATA[i];
    const imgPath = `./samples/${item.filename}`;

    try {
      const resp = await fetch(imgPath);
      const blob = await resp.blob();

      const { probs, time } = await runInference(blob);
      const { bestIndex, bestProb } = argmax(probs);
      const isCorrect = bestIndex === item.label;

      if (isCorrect) correct++;
      totalTime += time;
      totalConf += bestProb;

      const row = document.createElement('tr');
      row.className = isCorrect ? 'bg-green-50' : 'bg-red-50';
      row.innerHTML = `
        <td class="col-res text-center">${isCorrect ? '✅' : '❌'}</td>
        <td class="col-img">
          <div class="flex items-center gap-2">
            <img src="${imgPath}" class="w-10 h-10 object-cover rounded border cursor-pointer" onclick="openModal('${imgPath}')">
            <span class="text-xs text-gray-500">${getLabelName(item.label)}</span>
          </div>
        </td>
        <td class="text-xs">
          <div>正解: ${getLabelName(item.label)}</div>
          <div class="${isCorrect ? 'text-green-700' : 'text-red-700'} font-bold">予測: ${getLabelName(bestIndex)}</div>
        </td>
        <td class="col-conf text-right font-mono text-xs">
          ${(bestProb * 100).toFixed(1)}%
        </td>`;
      benchTableBody.prepend(row);

      const count = i + 1;
      benchAccEl.textContent = `${((correct / count) * 100).toFixed(1)}%`;
      benchTimeEl.textContent = `${(totalTime / count).toFixed(0)}ms`;
      benchConfEl.textContent = `${((totalConf / count) * 100).toFixed(1)}%`;
      benchProgEl.textContent = `${count}/${BENCHMARK_DATA.length}`;
      benchProgressBar.style.width = `${(count / BENCHMARK_DATA.length) * 100}%`;
    } catch (e) {
      console.error(`Error processing ${item.filename}:`, e);
    }

    if (i % 5 === 0) await new Promise(r => setTimeout(r, 0));
  }

  isRunning = false;
  enableRunButtons(true);
  benchProgressContainer.style.display = 'none';
}

// カスタム検証
function handleCustomFiles(input) {
  customImages.forEach(i => URL.revokeObjectURL(i.url));
  customImages = [];
  const container = document.getElementById('custom-staging');
  container.innerHTML = '';

  Array.from(input.files || []).forEach(file => {
    const url = URL.createObjectURL(file);
    const div = document.createElement('div');
    div.className = "flex gap-2 items-center border-b border-gray-200 py-2";

    const img = document.createElement('img');
    img.src = url;
    img.className = "thumb";
    img.onclick = () => openModal(url);

    const select = document.createElement('select');
    let options = `<option value="-1">正解ラベルを選択...</option>`;
    LABELS.forEach((_, idx) => options += `<option value="${idx}">[${idx}] ${getLabelName(idx)}</option>`);
    select.innerHTML = options;
    select.onchange = (e) => {
      const t = customImages.find(item => item.url === url);
      if (t) t.labelIndex = parseInt(e.target.value, 10);
    };

    div.appendChild(img);
    div.appendChild(select);
    container.appendChild(div);
    customImages.push({ file, labelIndex: -1, url });
  });

  document.getElementById('custom-result-panel').style.display = 'none';
  btnRunCustom.disabled = customImages.length === 0 || !session;
}

function clearCustom() {
  customImages.forEach(i => URL.revokeObjectURL(i.url));
  customImages = [];
  document.getElementById('custom-staging').innerHTML = '<p class="text-xs text-gray-400 text-center">ここにプレビューが表示されます</p>';
  document.getElementById('custom-result-panel').style.display = 'none';
  btnRunCustom.disabled = true;
}

async function runCustomBatch() {
  if (isRunning || customImages.length === 0 || !session) return;

  isRunning = true;
  enableRunButtons(false);
  customResultPanel.style.display = 'block';
  const tbody = document.getElementById('custom-table-body');
  tbody.innerHTML = '';

  let correct = 0;
  let totalTime = 0;
  let totalConf = 0;
  let countWithLabel = 0;

  for (let i = 0; i < customImages.length; i++) {
    const item = customImages[i];
    const { probs, time } = await runInference(item.file);
    const { bestIndex, bestProb } = argmax(probs);
    const isCorrect = item.labelIndex !== -1 && bestIndex === item.labelIndex;

    totalTime += time;
    totalConf += bestProb;
    if (item.labelIndex !== -1) countWithLabel++;
    if (isCorrect) correct++;

    const row = document.createElement('tr');
    row.className = isCorrect ? 'bg-green-50' : (item.labelIndex === -1 ? '' : 'bg-red-50');
    row.innerHTML = `
      <td class="col-res text-center">${item.labelIndex === -1 ? '－' : (isCorrect ? '✅' : '❌')}</td>
      <td class="col-img"><img src="${item.url}" class="thumb" onclick="openModal('${item.url}')"></td>
      <td class="text-xs">
        <div>指定: ${item.labelIndex === -1 ? '未指定' : getLabelName(item.labelIndex)}</div>
        <div class="${isCorrect ? 'text-green-700' : 'text-red-700'} font-bold">予測: ${getLabelName(bestIndex)}</div>
      </td>
      <td class="col-conf text-right font-mono text-xs">${(bestProb * 100).toFixed(1)}%</td>`;
    tbody.appendChild(row);
  }

  const total = customImages.length;
  document.getElementById('custom-count').textContent = `${total}`;
  document.getElementById('custom-time').textContent = `${(totalTime / total).toFixed(0)}ms`;
  document.getElementById('custom-conf').textContent = `${((totalConf / total) * 100).toFixed(1)}%`;
  document.getElementById('custom-acc').textContent = countWithLabel > 0 ? `${((correct / countWithLabel) * 100).toFixed(1)}%` : '--';

  isRunning = false;
  enableRunButtons(true);
}

// UI helpers
function renderDatasetGrid() {
  datasetGrid.innerHTML = '';
  BENCHMARK_DATA.slice(0, 200).forEach(item => {
    const div = document.createElement('div');
    div.className = "dataset-item";
    div.innerHTML = `
      <img src="./samples/${item.filename}" loading="lazy">
      <div class="dataset-label">${getLabelName(item.label)}</div>`;
    div.onclick = () => openModal(`./samples/${item.filename}`);
    datasetGrid.appendChild(div);
  });
}

function getLabelName(index) {
  const en = LABELS[index];
  if (en) return en;
  return `Unknown (${index})`;
}

function switchTab(tabId) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('[onclick="switchTab(\'' + tabId + '\')"]').forEach(t => t.classList.add('active'));
  document.getElementById('view-bench').classList.add('hidden');
  document.getElementById('view-dataset').classList.add('hidden');
  document.getElementById('view-custom').classList.add('hidden');
  document.getElementById(`view-${tabId}`).classList.remove('hidden');
}

function openModal(src) {
  document.getElementById('modal-img').src = src;
  document.getElementById('image-modal').classList.add('active');
}
function closeModal() {
  document.getElementById('image-modal').classList.remove('active');
}
