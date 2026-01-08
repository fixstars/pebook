#!/bin/sh
export MASTER_ADDR="192.168.1.1"
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=Init
export TIKTOKEN_CACHE_DIR=""
# export LD_LIBRARY_PATH=$PWD/nccl/build/lib:\${LD_LIBRARY_PATH}

TUNING_DIR=/net/efs8/data/project/performance-tuning/share/tuning-example


gbs={{global_batch_size}}
mbs={{micro_batch_size}}
pp={{pipeline_model_parallel_size}}
cp={{context_model_parallel_size}}
tp={{tensor_model_parallel_size}}
recompute_num_layers={{recompute_num_layers}}
empty_unused_memory_level={{empty_unused_memory_level}}

TRAIN_SCRIPT=/data/work/hiroaki.hosokawa/llm_book/scripts/tune/tuning_train_8b.sh
OUTPUT_DIR=

mpirun -x MASTER_ADDR \
    $TRAIN_SCRIPT \
    /data/work/hiroaki.hosokawa/llm_book/outputs/tuned \
    $HF_HOME/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/original/ \
    $gbs \
    $mbs \
    $pp \
    $cp \
    $tp \
    $recompute_num_layers \
    $empty_unused_memory_level \
