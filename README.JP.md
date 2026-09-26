# Open-LLM-VTuber-Rinne

凛祢とテキスト・音声で会話できるデスクトップアプリです。9種類の衣装、日記、長期記憶に対応しています。

現在のインストール手順は [README（中国語）](README.md) にまとめています。Windows 版は、次の順番で進めてください。

## 初めて使う場合

1. Git、uv、Ollama、7-Zip をインストールします。
2. プロジェクトを保存先のフォルダーにダウンロードします。C ドライブ以外も選べます。
3. `conf.yaml` に API キーを入力し、日記用のキーも設定します。
4. Ollama のモデルをダウンロードします。
5. GPT-SoVITS と凛祢の V2 音声モデル2点をダウンロードし、設定します。
6. 音声サービスとバックエンドを起動します。
7. デスクトップ版の `.exe` をインストールして開きます。

衣装はプロジェクトに含まれています。ゲームファイルを別途取り込む必要はありません。

コマンド、ファイルの配置場所、設定例は[インストール手順](README.md#prepare)を参照してください。

## インストール済みの場合

毎回の起動順は **Ollama → 音声サービス → バックエンド → デスクトップアプリ** です。

- [起動手順](README.md#daily-start)
- [旧版からの更新](README.md#upgrade)
- [トラブル対処](README.md#faq)

更新は元のプロジェクトフォルダーで行います。先に `conf.yaml` と `chat_history` をバックアップし、元の記憶を削除しないでください。

このプロジェクトは [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber) をベースにしています。凛祢の導入・更新には、上記の凛祢用ガイドを使用してください。

[中文](README.md) | [한국어](README.KR.md)

## 📜 サードパーティライセンス (Third-Party Licenses)

### Live2D サンプルモデルに関する通知 (Live2D Sample Models Notice)

このプロジェクトには、**Live2D Inc.から提供されたLive2Dサンプルモデル** が含まれています。当該資産は **Live2D Free Material License Agreement** および **Live2D Cubism Sample Data 利用規約** に基づき別途ライセンスが付与されており、このプロジェクトのMITライセンスには含まれません。

このコンテンツはLive2D Inc.が所有し著作権を持つサンプルデータを使用しており、Live2D Inc.が定めた **規約と条件** に従って活用されます。（詳細は [Live2D Free Material License Agreement](https://www.live2d.jp/en/terms/live2d-free-material-license-agreement/) および [Terms of Use](https://www.live2d.com/eula/live2d-sample-model-terms_en.html) を参照）

注：特に中堅・大規模企業での **商用利用** の際、このLive2Dサンプルモデルの使用には追加のライセンス要件が適用される場合があります。プロジェクトを商用利用する計画がある場合は、必ずLive2D Inc.から適切な許可を得るか、当該モデルが含まれていないバージョンを使用してください。
