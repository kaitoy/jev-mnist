# jev-mnist

TypeSafe AI の **Jev 1.14**（画像入力対応）に MNIST の手書き数字画像を見せて 0〜9 のどれかを答えさせ、正解率を計測するプログラムです。

- 画像は 28x28 のグレースケールを PNG にし（既定で 4 倍に拡大）、`typesafe_sdk.Image` として `state` に渡します。
- 質問は `Choice`（選択肢 `"0"`〜`"9"`）で、Jev が最も確率の高い数字を返します。
- 既定モデルは `jev-preview`（Jev 1.14.0 を指すエイリアス）。`--model` で変更できます。

> Jev 1.14 の画像入力は TypeSafe 社が「実験的プレビュー」と位置づけている機能です。

## セットアップ

```sh
python -m venv .venv
.venv/bin/pip install -e '.[test]'
export TYPESAFE_API_KEY=...   # TypeSafe AI の API キー
```

## 実行

```sh
# テストセットの先頭 100 枚で評価（既定）
.venv/bin/jev-mnist

# ランダムに 500 枚、16 並列、結果を CSV に保存
.venv/bin/jev-mnist --limit 500 --seed 0 --concurrency 16 --output results.csv

# API を呼ばずに、送信する画像を data/samples/ に保存して確認
.venv/bin/jev-mnist --dry-run --limit 10
```

MNIST テストセット（10,000 枚）は初回実行時に `data/` へダウンロードされます。
出力には正解率（応答できた分／全体）、数字ごとの正解率、混同行列、入力トークン数、実際に応答したモデル名が表示されます。
API エラーになった画像は「errors」として数え、処理は止めません。

| オプション | 既定値 | 説明 |
|---|---|---|
| `--limit` | 100 | 評価する画像の枚数 |
| `--offset` | 0 | 先頭から評価する場合の開始位置 |
| `--seed` | なし | 指定するとランダムに `--limit` 枚を抽出 |
| `--model` | `jev-preview` | 使用するモデル |
| `--concurrency` | 8 | 同時リクエスト数 |
| `--upscale` | 4 | 画像の拡大倍率 |
| `--timeout` | 60 | リクエストごとのタイムアウト（秒） |
| `--output` | なし | 画像ごとの結果を書き出す CSV |
| `--dry-run` | - | API を呼ばず画像だけ保存 |

## テスト

```sh
.venv/bin/pytest
```

API キーやネットワークなしで動くオフラインテスト（MNIST パーサ、PNG 変換、集計）です。
