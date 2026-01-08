#!/bin/bash -x

# [I 2025-10-22 07:27:14,341] Trial 57 finished with value: 607.4247746403573 and parameters: {'tensor_model_parallel_size
# _index': 1, 'pipeline_model_parallel_size_index': 2, 'context_parallel_size_index': 0, 'micro_batch_size_index': 0, 'fp'
# : 'fp8', 'recompute_granularity': 'full', 'recompute_method': 'uniform', 'recompute_num_layers_index': 1, 'empty_unused_
# memory_level': 0}. Best is trial 57 with value: 607.4247746403573.

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=14535 # 適当な値
export LOCAL_RANK=$((OMPI_COMM_WORLD_RANK % GPUS_PER_NODE))

GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
TP=2
PP=4
CP=1

CHECKPOINT_LOAD_PATH=$1
CHECKPOINT_SAVE_PATH=$2
TOKENIZER_SAVE_PATH=$3
DATA_PATH_BASE=$4
DATA_PATH=$(ls -1 $DATA_PATH_BASE/*.bin | tr '\n' ' ' | sed s/\.bin//g)
DATA_CACHE_PATH=./cache
TENSORBOARD_LOGS_PATH=./tensorboard

GPT_MODEL_ARGS=(
    --num-layers 80
    --hidden-size 8192
    --num-attention-heads 64
    --seq-length 8192
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --max-position-embeddings 131072
    --use-rope-scaling
    --rotary-base 500000
    --rotary-percent 1.0
    --attention-dropout 0
    --hidden-dropout 0
    --normalization RMSNorm
    --ffn-hidden-size 28672
    --num-query-groups 8
    --swiglu
    --group-query-attention
    --tokenizer-type HuggingFaceTokenizer
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear
    --tokenizer-model $TOKENIZER_SAVE_PATH
    --no-load-optim
    --no-load-rng
    --exit-on-missing-checkpoint
    --use-mcore-models
    --attention-backend fused
    --ckpt-format torch
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 20
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr-decay-style cosine
    --lr 1.8e-6
    --min-lr 1.8e-7
    --lr-warmup-fraction .025
    --lr-decay-iters 39000
    --use-flash-attn
    --empty-unused-memory-level 0
    --recompute-granularity "full"
    --recompute-method "uniform"
    --recompute-num-layers 2
    --transformer-impl "transformer_engine"
    --optimizer adam
    --use-distributed-optimizer
    --fp8-format hybrid
)

MODEL_PARALLEL_ARGS=(
    --sequence-parallel
    --tensor-model-parallel-size $TP
    --context-parallel-size $CP
    --pipeline-model-parallel-size $PP
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 949,44,7
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

# ToDo: fix
# source ./.venv/bin/activate
source ../.venv/bin/activate

python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

