# 第05章のサンプルコード

本章では、大規模言語モデル Llama-3.2-1B のフルパラメータ事後学習（指示チューニング）に対してパフォーマンスエンジニアリングを実践します。

## ファイル構成

### ベーススクリプト（DeepSpeedExamples）

本章は [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples/tree/bd47e5b/training/tensor_parallel) を題材にしています。

| ファイル名 | 説明 |
|------------|------|
| `train.py` | 元実装の学習スクリプト |
| `run.sh` | 学習実行スクリプト |
| `utils.py` | ユーティリティ関数 |
| `alpaca_data.json` | Alpacaデータセット |
| `configs/ds_config_temp.json` | DeepSpeed設定ファイルのテンプレート |
| `configs/ds_config.json` | DeepSpeed設定ファイル（run.shで自動生成） |
| `requirements.txt` | 依存パッケージ |
| `calculate_dataset_result_tokens.py` | データセットのトークン数計算スクリプト |

### 各節で使用するスクリプト

| ファイル名 | 対応する節 | 説明 |
|------------|------------|------|
| `train_with_profiler.py` | 5.2節〜5.3節 | プロファイラ導入版（MemoryCallback有効） |
| `train_without_memcallback.py` | 5.5節 | MemoryCallback削除版（環境変数で制御） |
| `optimize.py` | 5.6節 | Optunaによるハイパーパラメータ自動チューニング |
| `run_optimize_template.sh` | 5.6節 | チューニング用スクリプトテンプレート |
| `configs/ds_config_optimize_template.json` | 5.6節 | チューニング用DeepSpeed設定テンプレート |
| `analyze_study.py` | 5.6節 | チューニング結果の可視化 |

## 実行環境

*   NVIDIA A100 40GB GPU × 4台
*   NVIDIA GPU Cloud (NGC) の PyTorch イメージ (`nvcr.io/nvidia/pytorch:25.08-py3`)

### 環境構築

```bash
docker run \
    -it \
    --rm \
    --volume `pwd`:/work \
    --workdir /work \
    --gpus all \
    --ipc=host \
    --net=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --env HF_HOME=. \
    nvcr.io/nvidia/pytorch:25.08-py3
```

コンテナ内で依存パッケージをインストール:

```bash
pip install -r requirements.txt
```

### モデルのダウンロード

事前に Hugging Face にログインし、Llama モデルの利用規約に同意した上で、Llama-3.2-1B をダウンロードします。

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-1B
```

## 各節の実行方法

### 5.1節〜5.2節: 初期状態の把握

学習は以下のコマンドで実行します。

```bash
bash run.sh
```

### 5.3節: プロファイラの導入

`run.sh` の中で呼び出すスクリプトを `train_with_profiler.py` に変更します。

```diff
- deepspeed --num_gpus $num_gpus  \
-     --master_port 51336  train.py  \
+ deepspeed --num_gpus $num_gpus  \
+     --master_port 51336  train_with_profiler.py  \
```

Nsight Systems でプロファイリングする場合:

```bash
PROFILE_NSYS=1 nsys profile \
    -t cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    bash run.sh
```

PyTorch Profiler でプロファイリングする場合:

```bash
PROFILE_TORCH=1 bash run.sh
```

### 5.4節: MemoryCallbackの削除

`run.sh` の中で呼び出すスクリプトを `train_without_memcallback.py` に変更します。

```diff
- deepspeed --num_gpus $num_gpus  \
-     --master_port 51336  train.py  \
+ deepspeed --num_gpus $num_gpus  \
+     --master_port 51336  train_without_memcallback.py  \
```

メモリ使用量を観測したい場合は環境変数を設定:

```bash
PROFILE_MEM=1 bash run.sh
```

### 5.5節: ハイパーパラメータ自動チューニング

以下のコマンドで Optuna によるハイパーパラメータ自動チューニングを実行します。

```bash
python optimize.py
```

以下のコマンドでチューニング結果を可視化します。

```bash
python analyze_study.py
```

### 5.6節: Torch Compileの適用

`run.sh` に以下のオプションを追加します:

```diff
  deepspeed --num_gpus $num_gpus  \
      --master_port 51336  train_without_memcallback.py  \
      ...
+     --torch_compile True \
+     --torch_compile_backend inductor \
      --deepspeed "./configs/ds_config.json"
```

### 5.7節: グローバルバッチサイズの調整

`run.sh` 内の以下のパラメータを調整します:

*   `--per_device_train_batch_size`: マイクロバッチサイズ
*   `--gradient_accumulation_steps`: 勾配累積段数
