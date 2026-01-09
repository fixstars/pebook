# 第04章のサンプルコード

## ハードウェア要件

本スクリプトは[さくらインターネット高火力phy(H100)](https://www.sakura.ad.jp/koukaryoku-phy/)で動作確認を実施しています。

- CUDA version: 12.8
- CUDA device name: NVIDIA H100 80GB HBM3

## 実行環境の作成

実行するためにはuvをインストールする必要があります。<https://github.com/astral-sh/uv> を参考に、uvをインストールしてください。
uvをインストールした後に以下のコマンドを実行して、実行環境を作成してください。

- PyTroch version: 2.8.0+cu128
- flash attn version: 2.8.3

```shell
cd chapter04/src
uv sync
uv sync --group flash-attn
```

## モデルダウンロード

本スクリプトで利用するモデルはGated modelとなっています。

- `meta-llama/Meta-Llama-3-8B-Instruct`
  - モデルURL: <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>
- `meta-llama/Llama-3.2-1B-Instruct`
  - モデルURL: <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>

自身のHugging Faceアカウントで使用申請をした上で、ローカルにモデルをダウンロードして下さい。

参考: <https://huggingface.co/docs/hub/models-downloading>

## データセットダウンロード

モデルダウンロードと同様に、ローカルにモデルをダウンロードして下さい。こちらは、使用申請は不要です。

- `llm-jp/Synthetic-JP-EN-Coding-Dataset`
  - データセットURL: <https://huggingface.co/datasets/llm-jp/Synthetic-JP-EN-Coding-Dataset>

## 実行(transformers)

以下のコマンドで実行できます。
```shell
uv run llama3-inference.py
```

以下の引数を指定することができます。
  - `--model-name` (str, デフォルト: meta-llama/Meta-Llama-3-8B-Instruct)
    - メインモデル名
  - `--draft-name` (str, デフォルト: None)
    - 投機的デコーディングで用いるドラフトモデル名
  - `--device-map` (str, デフォルト: auto)
  - `--num-samples` (int, デフォルト: 128)
    - 処理するサンプルの数。
repeat=1`）。
  - `--batch-size` (int, デフォルト: 1)
    - 推論時のバッチサイズ。
  - `--max-new-tokens` (int, デフォルト: 1000)
    - サンプルごとに生成される最大トークン数。
  - `--random-prompt-length` (int, デフォルト: 0)
    - データセットの代わりにランダム文字列で任意のシーケンス長のプロンプトを使用します。
  - `--simple-prompt` (str, デフォルト: None)
    - データセットの代わりに全サンプルで同じ簡易プロンプトを使用します。
  - `--num_assistant_tokens` (int, デフォルト: 20)
    - 投機的デコーディングのドラフト出力長。
  - `--use-quantize` (flag, デフォルト: False)
    - 4-bit量子化を有効にします（BitsAndBytes）。
  - `--use_int8` (flag, デフォルト: False)
    - 8-bit量子化を有効にします（bitsandbytes）。
  - `--use-flash-attention-2` (flag, デフォルト: False)
    - Flash Attention 2を有効にします。
  - `--use-speculative-decoding` (flag, デフォルト: False)
    - Llama3 8B+Llama3.2 1Bの投機的デコーディングを有効にします。
  - `--use-large-speculative-decoding` (flag, デフォルト: False)
    - Llama3 70B+Llama3 Bの投機的デコーディングを有効にします。
  - `--use-profiler` (flag, デフォルト: False)
    - PyTorchプロファイラーを有効にします。
  - `--record-shapes` (bool, デフォルト: False)
    - Tensor の形状記録を有効にします。
  - `--profile-memory` (bool, デフォルト: False)
    - メモリプロファイリングを有効にします。
  - `--with-stack` (bool, デフォルト: True)
    - スタックトレース記録を有効にします。
  - `--with-flops` (bool, デフォルト: False)
    - FLOPs 推定を有効にします。
  - `--with-modules` (bool, デフォルト: False)
    - モジュール情報の記録を有効にします。

単純なプロンプトを試したい場合は以下のように実行することで試すことができます。

```shell
uv run llama3-inference.py --simple-prompt "Hello, Who are you?"
```

大きいtraceを開く場合はperfettoのサーバを建て、ポートフォワードすることで大きいファイルのやり取りをせずに確認することができます。

```shell
curl -LO https://get.perfetto.dev/trace_processor
chmod +x ./trace_processor
./trace_processor --httpd /path/to/XXXX.pt.trace.json
# Port Forwarding: 9001 -> 9001
# Open: https://ui.perfetto.dev/#!/?rpc_port=9001
```

## 実行(vLLM)

以下のコマンドで実行できます。
```shell
uv run llama3-inference-vllm.py
```

以下の引数を指定することができます。
  - `--quantization` (str, デフォルト: None)
    - 適用する量子化の種類
  - `--num-samples` (int, デフォルト: 128)
    - 処理するサンプルの数。
repeat=1`）。
  - `--max-new-tokens` (int, デフォルト: 1000)
    - サンプルごとに生成される最大トークン数。
  - `--simple-prompt` (str, デフォルト: None)
    - データセットの代わりに全サンプルで同じ簡易プロンプトを使用します。
  - `--tensor-parallel-size` (int, デフォルト: 1)
    - テンソル並列サイズ（＝使用するGPU数）

## 本書での実際の実行コマンド

### ベースライン実行

```shell
uv run llama3-inference.py \
    --batch-size 1 \
    --num-samples 512 \
    --max-new-tokens 1000 \
    --device-map auto
```

### ベースラインのプロファイリング

```shell
uv run llama3-inference.py \
    --batch-size 1 \
    --num-samples 4 \
    --max-new-tokens 1000 \
    --device-map auto \
    --use-profiler
```

### 1GPU・2GPUでの利用

```shell
# 1GPUでの実行
uv run llama3-inference.py \
    --batch-size 1 \
    --num-samples 512 \
    --max-new-tokens 1000 \
    --device-map cuda:0
# 2GPUでの実行
CUDA_VISIBLE_DEVICES=0,1 uv run llama3-inference.py \
    --batch-size 1 \
    --num-samples 512 \
    --max-new-tokens 1000 \
    --device-map auto
```


### FlashAttention 2の適用

```shell
uv run llama3-inference.py \
    --batch-size 1 \
    --num-samples 512 \
    --max-new-tokens 1000 \
    --device-map cuda:0 \
    --use-flash-attention-2
```

### 重みの量子化の適用

```shell
uv run llama3-inference.py \
    --batch-size 1 \
    --num-samples 512 \
    --max-new-tokens 1000 \
    --device-map cuda:0 \
    --use-quantize
```

### 70Bモデルでの実行

```shell
CUDA_VISIBLE_DEVICES=0,1 uv run llama3-inference.py \
    --batch-size 1 \
    --device-map auto \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct
```

### 投機的デコーディングでの実行

```shell
CUDA_VISIBLE_DEVICES=0,1 uv run llama3-inference.py
    --batch-size 1 \
    --device-map auto \
    --model-name meta-llama/Meta-Llama-3-70B-Instruct \
    --draft-name meta-llama/Meta-Llama-3-8B-Instruct \
    --use-speculative-decoding \
    --num-assistant-tokens 20
```

### vLLMでの推論

```shell
# 通常の推論
uv run llama3-inference-vllm.py --num-samples 512
# FP8量子化
uv run llama3-inference-vllm.py --num-samples 512 --quantization fp8
```
