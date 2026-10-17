# jev-mnist

TypeSafe AI の **Jev 1.14**（画像入力対応）に MNIST の手書き数字画像を見せて 0〜9 のどれかを答えさせ、正解率を計測するプログラムです。

- 画像は 28x28 のグレースケールを PNG にし（既定で 10 倍に拡大して 280x280）、`typesafe_sdk.Image` として `state` に渡します。
- 質問は `Choice`（選択肢 `"0"`〜`"9"`）で、Jev が最も確率の高い数字を返します。
- 既定モデルは `jev-preview`（Jev 1.14.0 を指すエイリアス）。`--model` で変更できます。

> Jev 1.14 の画像入力は TypeSafe 社が「実験的プレビュー」と位置づけている機能です。

## セットアップ

[mise](https://mise.jdx.dev/) で Python と [uv](https://docs.astral.sh/uv/) を用意し、uv で依存パッケージをインストールします（バージョンは `mise.toml`、依存のロックは `uv.lock`）。

```sh
mise install                  # mise.toml の Python と uv をインストール
uv sync --extra test          # uv.lock どおりに .venv へインストール
cp .env.example .env          # .env に TypeSafe AI の API キーを書く
```

API キーは `.env` の `TYPESAFE_API_KEY` に書いておけば起動時に読み込まれます（`.env` は Git 管理外）。
環境変数 `TYPESAFE_API_KEY` が設定されている場合はそちらが優先されます。

## 実行

```sh
# テストセットの先頭 100 枚で評価（既定）
uv run jev-mnist

# ランダムに 500 枚、16 並列、結果を CSV に保存
uv run jev-mnist --limit 500 --seed 0 --concurrency 16 --output results.csv

# API を呼ばずに、送信する画像を data/samples/ に保存して確認
uv run jev-mnist --dry-run --limit 10
```

MNIST テストセット（10,000 枚）は初回実行時に `data/` へダウンロードされます。
出力には正解率（応答できた分／全体）、数字ごとの正解率、混同行列、入力トークン数、実際に応答したモデル名、判定速度（1 枚あたりの応答時間の平均・中央値・p95・最小・最大と、全体の所要時間・スループット）が表示されます。
`--output` の CSV には画像ごとの応答時間が `latency_ms` 列（ミリ秒）として入ります。
API エラーになった画像は「errors」として数え、処理は止めません。

| オプション | 既定値 | 説明 |
|---|---|---|
| `--limit` | 100 | 評価する画像の枚数 |
| `--offset` | 0 | 先頭から評価する場合の開始位置 |
| `--seed` | なし | 指定するとランダムに `--limit` 枚を抽出 |
| `--model` | `jev-preview` | 使用するモデル |
| `--concurrency` | 8 | 同時リクエスト数 |
| `--upscale` | 10 | 画像の拡大倍率（API は 1 辺 280〜1568px のみ受け付けるため 10〜56） |
| `--timeout` | 60 | リクエストごとのタイムアウト（秒） |
| `--output` | なし | 画像ごとの結果を書き出す CSV |
| `--dry-run` | - | API を呼ばず画像だけ保存 |

## 間違えた画像を見る

`--output` で保存した CSV から、Jev が間違えた画像だけを HTML ギャラリーにまとめます（API は呼びません）。

```sh
.venv/bin/jev-mnist --limit 500 --seed 0 --output results.csv
.venv/bin/jev-mnist-errors results.csv                         # 間違えた画像すべて → errors.html
.venv/bin/jev-mnist-errors results.csv --label 4 --prediction 9  # 正解 4 を 9 と答えたものだけ
.venv/bin/jev-mnist-errors results.csv --min-confidence 0.9      # 自信満々で間違えたものだけ
```

既定では確信度の高い順に並びます。主なオプション: `--label`, `--prediction`, `--min-confidence`, `--max-confidence`,
`--include-errors`（API エラーの画像も含める）, `--sort index`, `--limit`, `--output`（既定 `errors.html`）。

## テスト

```sh
uv run pytest
```

API キーやネットワークなしで動くオフラインテスト（MNIST パーサ、PNG 変換、集計）です。
