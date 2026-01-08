#!/bin/bash -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=14535 # 適当な値
export LOCAL_RANK=$((OMPI_COMM_WORLD_RANK % GPUS_PER_NODE))

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16

MEGATRON_PATH=/data/work/hiroaki.hosokawa/llm_book/Megatron-LM

cd $MEGATRON_PATH

CHECKPOINT_LOAD_PATH=/data/work/hiroaki.hosokawa/llm_book/checkpoints/Llama-3.1-8B-mega
CHECKPOINT_SAVE_PATH=/data/work/hiroaki.hosokawa/llm_book/outputs/8B
TOKENIZER_SAVE_PATH=$HF_HOME/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/
TENSORBOARD_LOGS_PATH=/data/work/hiroaki.hosokawa/llm_book/outputs/8B/tensorboard
DATA_PATH_BASE=/data/work/hiroaki.hosokawa/llm_book/datasets
DATA_PATH=$(ls -1 $DATA_PATH_BASE/*.bin | tr '\n' ' ' | sed s/\.bin//g)
DATA_CACHE_PATH=/data/work/hiroaki.hosokawa/llm_book/cache

GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32
    --seq-length 4096
    # --no-position-embedding #
    --no-masked-softmax-fusion #
    --attention-softmax-in-fp32
    # --use-rotary-position-embeddings #
    --max-position-embeddings 131072
    --use-rope-scaling
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-dropout 0
    --hidden-dropout 0
    --normalization RMSNorm #
    --ffn-hidden-size 14336
    --num-query-groups 8
    --swiglu
    --group-query-attention
    --tokenizer-type HuggingFaceTokenizer
    # --tokenizer-type Llama3Tokenizer
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear
    --tokenizer-model $TOKENIZER_SAVE_PATH
    --no-load-optim #
    --no-load-rng #
    # --finetune
    # --auto-detect-ckpt-format
    # --override-opt_param-scheduler
    --exit-on-missing-checkpoint
    # --kv-channels 128
    # --use-checkpoint-args
    --use-mcore-models
    --attention-backend fused
    --ckpt-format torch
    # --apply-layernorm-1p
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 39000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    # --fp16
    --bf16
    --lr-decay-style cosine
    --lr 1.8e-6
    --min-lr 1.8e-7
    --lr-warmup-fraction .025
    --lr-decay-iters 39000
    --use-flash-attn
    # --loss-scale 16384
    # --fp8-format 'hybrid'
    # --recompute-activations
    # --recompute-granularity "selective"
    --recompute-granularity "full"
    --recompute-method "block"
    --recompute-num-layers 3
    --transformer-impl "transformer_engine"
    --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather
)

MODEL_PARALLEL_ARGS=(
    --sequence-parallel
    --tensor-model-parallel-size 1
    --context-parallel-size 1
    --pipeline-model-parallel-size 1
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 949,44,7
    # --no-create-attention-mask-in-dataloader
    # --no-mmap-bin-files
    # --split 920,40,40
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 250
    --eval-interval 5000
    --save $CHECKPOINT_SAVE_PATH
    --load $CHECKPOINT_LOAD_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-throughput
)

PROFILE_ARGS=(
    --profile
    --profile-step-start 3
    --profile-step-end 4
    # --profile-ranks "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31"
    --profile-ranks "0 1 2 3"
)

source /data/work/hiroaki.hosokawa/llm_book/.venv/bin/activate

python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

