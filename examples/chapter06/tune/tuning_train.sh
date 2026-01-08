#!/bin/bash


# Prevents GPU processes becoming zombies on unexpected termination
KILL_ALL_GPU_PROCESSES() {
	sleep 5
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -tr kill
}
KILL_ALL_GPU_PROCESSES
trap KILL_ALL_GPU_PROCESSES EXIT

export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_FUSED_ATTN_BACKEND=2 # FP8 attention calculations by setting to 2
export TORCH_NCCL_AVOID_RECORD_STREAMS=1


export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
export RANK=$OMPI_COMM_WORLD_RANK
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=14535 # 適当な値
export LOCAL_RANK=$((OMPI_COMM_WORLD_RANK % GPUS_PER_NODE))

echo $WORLD_SIZE

MICRO_BATCH_SIZE=$4
GLOBAL_BATCH_SIZE=$3

TP=$7
PP=$5
CP=$6

PROJECT_ROOT=/data/work/hiroaki.hosokawa/llm_book
MEGATRON_PATH=$PROJECT_ROOT/Megatron-LM
CHECKPOINT_LOAD_PATH=$1
CHECKPOINT_SAVE_PATH=$1
TOKENIZER_SAVE_PATH=$2
TENSORBOARD_LOGS_PATH=$1/tensorboard
DATA_PATH_BASE=$PROJECT_ROOT/datasets
DATA_PATH=$(ls -1 $DATA_PATH_BASE/*.bin | tr '\n' ' ' | sed s/\.bin//g)
DATA_CACHE_PATH=$PROJECT_ROOT/cache

GPT_MODEL_ARGS=(
    --num-layers 80
    --hidden-size 8192
    --num-attention-heads 64
    --seq-length 8192
    --no-masked-softmax-fusion
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
    --train-iters 5
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
    --use-flash-attn
    --empty-unused-memory-level ${11}
    --transformer-impl "transformer_engine"
    --use-distributed-optimizer
)
if [[ "$9" != "None" ]]; then
    TRAINING_ARGS+=(--recompute-granularity "$9")
fi
if [[ "${10}" != "None" ]]; then
    TRAINING_ARGS+=(--recompute-method ${10})
fi
if [[ "$8" != "0" ]]; then
    TRAINING_ARGS+=(--recompute-num-layers $8)
fi
if [[ "${12}" == "fp8" ]]; then
    TRAINING_ARGS+=(--fp8-format hybrid)
fi

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size ${TP}
	--pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
	--sequence-parallel
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-throughput
)

cd $MEGATRON_PATH
source $PROJECT_ROOT/.venv/bin/activate

# 5 iter 600 秒以上かかる試行はタイムアウト
timeout -sKILL 600 python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
