# LMFRNet Benchmark (Local)

このリポジトリをクローンして、PC上でローカルに実行するためのガイドです。

## 準備

まず、リポジトリをダウンロード（クローン）します。

```bash
git clone https://github.com/LokiXps/lmfrnet-benchmark-private.git
cd lmfrnet-benchmark-private
```

## 使い方 1: Webアプリ版 (GUI)

ブラウザ上でグラフィカルに操作したい場合です。
セキュリティ制限（CORS）のため、HTMLを直接開くのではなく、簡易サーバー経由で開く必要があります。

1. `public` フォルダに移動します。
   ```bash
   cd public
   ```
2. ローカルサーバーを起動します。
   ```bash
   python3 -m http.server 8080
   ```
3. ブラウザで以下のURLを開きます。
   [http://localhost:8080](http://localhost:8080)

## 使い方 2: Pythonスクリプト版 (CUI)

コマンドラインで直接推論を実行したい場合です。

1. 必要なライブラリをインストールします。
   ```bash
   pip install onnxruntime numpy pillow
   ```
2. スクリプトを実行します。
   ```bash
   # 精度検証
   python3 scripts/verify_accuracy.py
   ```
