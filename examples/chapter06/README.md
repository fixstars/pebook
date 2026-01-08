# 第06章のサンプルコード

Megatron-LM による Llama3.1-70B の事前学習を動作させ、また ZenithTune によるパラメータチューニングを実行します

## 前提条件

- 環境
    - NVIDIA driver 570.172.08
    - CUDA 12.6
    - CUDNN 9
	    - インストールされていない場合、`sudo apt-get -y install cudnn9-cuda-12` でインストール
    - OpenMPI 5.0.5
	    - 複数ノードで mpirun を実行できるように、パスフレーズなし鍵認証での SSH 接続をノード間で行えるようにしておく
    - Python 3.12
        - python3-dev も必要: `sudo apt install python3-dev`

## 環境構築

### venv

```bash
# clone & patch megatron-lm
cd chapter05/src/
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.3
patch -p1 < ../sgd.patch
pip install -e .

# install prerequisites
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install pybind11 flit wheel nltk tensorboard flask wandb mpi4py ninja cmake
pip install sgd-sai six psutil transformers accelerate "numpy<2.0"

# export envs
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export CUDNN_PATH=/usr/lib/x86_64-linux-gnu
export CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu
export NVTE_CUDA_INCLUDE_PATH=/usr/local/cuda/include

# TE / apex
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git@25.08

# install flash attention for hopper
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout af314d400663fe895199b0586a9f1f718b1d7b79
cd hopper
pip install .
python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flashattn_hopper
cp flash_attn_interface.py $python_path/flashattn_hopper

# Install Megatron-LM
pip install -e .

# install tools for tuning
pip install aibooster
```

## モデル準備

モデルをダウンロードし、Megatron 形式のチェックポイントに変換する

```bash
hf download meta-llama/Llama-3.1-70B

export CUDA_DEVICE_MAX_CONNECTIONS=1
python tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 2 \
    --checkpoint-type hf \
    --load-dir /path/to/models/Llama-3.1-70B \
    --save-dir ./checkpoints/Llama-3.1-70B-mega \
    --tokenizer-model /path/to/models/Llama-3.1-70B \
    --model-size llama3
```

もし `torch.AcceleratorError: CUDA error: initialization error` といったエラーが出た場合、以下のパッチを適用して再実行する

```bash
patch -p1 < ../fix_convert.patch
```

## データセット準備

データセットをダウンロードし前処理する

```bash
mkdir datasets
wget https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/arxiv_024de5df-1b7f-447c-8c3a-51407d8d6732.jsonl -O datasets/arxiv.jsonl

python tools/preprocess_data.py \
    --input ./datasets/arxiv.jsonl \
    --output-prefix ./datasets/llama3_arxiv \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/models/Llama-3.1-70B \
    --workers 64 \
    --append-eod
```

## 学習

2 ノードでの学習を行う

```bash
/opt/openmpi/bin/mpirun -x MASTER_ADDR=192.168.1.1 --host 192.168.1.1:120,192.168.1.2:120 --map-by ppr:8:node --cpus-per-rank 15 --mca plm_rsh_args '-o StrictHostKeyChecking=no -p 53412' \
    ../train_70b.sh \
    ./checkpoints/Llama-3.1-70B-mega \
    ./checkpoints_save/Llama-3.1-70B-mega \
    /path/to/models/Llama-3.1-70B \
    ./datasets
```

### 備考

optimizer に SGD を用いる場合は `--use-distributed-optimizer` オプションを外す。
Adam を用いる場合は、`--use-distributed-optimizer` オプションをつけ、`--optimizer adam` を指定する。

## パラメータチューニング

```bash
src/tune/run_tune.sh
```

## プロファイリング

### nsys

```diff
+ nsys profile -s none -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
    python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
-    ${EVAL_AND_LOGGING_ARGS[@]}
+    ${EVAL_AND_LOGGING_ARGS[@]} \
+    --profile --profile-step-start 10 --profile-step-end 12
```

カレントディレクトリに`nsys-rep`が生成されます．

### torchprof

```diff
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
-    ${EVAL_AND_LOGGING_ARGS[@]}
+    ${EVAL_AND_LOGGING_ARGS[@]} \
+    --profile --use-pytorch-profiler --profile-step-start 10 --profile-step-end 11
```

`./tensorboard`にjsonが生成されます．

`RuntimeError: Can't disable Kineto profiler when it's not running`のようなエラーが出る場合がありますが，プロファイリングには成功しています．
