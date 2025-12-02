# LMFRNet Benchmark (Local)

このリポジトリをクローンして、PC上でローカルに実行するためのガイドです。

## 準備

まず、リポジトリをダウンロード（クローン）します。

```bash
git clone https://github.com/LokiXps/lmfrnet-benchmark-private.git
cd lmfrnet-benchmark-private
```

## 使い方 (簡単)

フォルダ内の起動用ファイルをダブルクリックするだけで実行できます。

### 1. Webアプリ版 (GUI)
ブラウザでグラフィカルに操作します。

- **Windows**: `start_web_app.bat` をダブルクリック
- **Mac/Linux**: `start_web_app.sh` を実行

自動的にサーバーが立ち上がり、ブラウザが開きます。

### 2. Pythonスクリプト版 (CUI)
コマンドラインで推論を実行します（事前に `pip install onnxruntime numpy pillow` が必要です）。

- **Windows**: `run_benchmark.bat` をダブルクリック
- **Mac/Linux**: `run_benchmark.sh` を実行
