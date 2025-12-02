# セットアップガイド

## 概要
このアプリケーションは、ブラウザ上で動作する画像認識ベンチマークツールです。
ONNX Runtime WebAssemblyを使用しており、正しく動作させるためには特定のHTTPヘッダー（COOP/COEP）とMIMEタイプの設定が必要です。

## 起動方法

### Windows
フォルダ内の `start_web_app.bat` をダブルクリックしてください。

### Mac / Linux
ターミナルで以下のコマンドを実行するか、`start_web_app.sh` を実行してください。

```bash
./start_web_app.sh
```

### 手動起動
もしスクリプトが動かない場合は、以下の手順でサーバーを起動してください。

1. ターミナル（コマンドプロンプト）を開く
2. `public` フォルダに移動
3. 以下のコマンドを実行
   ```bash
   python3 server.py
   ```
4. ブラウザで `http://localhost:8080` にアクセス

## トラブルシューティング

### "Importing a module script failed" エラーが出る場合
ブラウザが `.mjs` ファイルを正しいMIMEタイプ（`application/javascript`）で読み込めていない可能性があります。
必ず `python3 -m http.server` ではなく、同梱の `server.py` を使用して起動してください。

### "SharedArrayBuffer is not defined" エラーが出る場合
セキュリティヘッダー（COOP/COEP）が不足しています。
これも `server.py` を使用することで解決します。
