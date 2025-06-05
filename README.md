# XPhoneBERT テストプロジェクト

[VinAI Research](https://github.com/VinAIResearch/XPhoneBERT)が公開しているXPhoneBERTの動作確認用のテストプロジェクトです。

## 概要

XPhoneBERTは、テキスト読み上げ（TTS）のための音素表現を学習した多言語事前学習モデルです。このプロジェクトでは、公式実装の基本的な機能をテストするための最小限の実装を提供しています。

## 環境要件

- Python 3.12
- CUDA対応GPU（推奨）

## セットアップ

1. 依存パッケージのインストール：
```bash
pip install -r requirements.txt
```

## 使用方法

基本的なテスト実行：
```bash
python run_xphonebert.py
```

オプション付きの実行例：
```bash
# カスタムテキストの処理
python run_xphonebert.py --text "こんにちは 、 世界 です 。"

# 英語テキストの処理
python run_xphonebert.py --text "Hello , world ." --language eng-us

# 特徴量の保存
python run_xphonebert.py --output-file features.pt

# CPUの強制使用
python run_xphonebert.py --force-cpu
```

## 注意事項

- このプロジェクトは公式実装のテスト用です
- 入力テキストは事前に単語分割されている必要があります
- 詳細な使用方法や機能については[公式リポジトリ](https://github.com/VinAIResearch/XPhoneBERT)を参照してください

## ライセンス

MIT License

## 引用

XPhoneBERTを使用する場合は、以下の論文を引用してください：

```
@inproceedings{xphonebert,
    title     = {{XPhoneBERT: A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech}},
    author    = {Linh The Nguyen and Thinh Pham and Dat Quoc Nguyen},
    booktitle = {Proceedings of the 24th Annual Conference of the International Speech Communication Association (INTERSPEECH)},
    year      = {2023}
}
```