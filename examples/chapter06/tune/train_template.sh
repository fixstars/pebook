#!/bin/sh
export MASTER_ADDR="192.168.1.1"
export TIKTOKEN_CACHE_DIR=""

gbs={{global_batch_size}}
mbs={{micro_batch_size}}
pp={{pipeline_model_parallel_size}}
cp={{context_parallel_size}}
tp={{tensor_model_parallel_size}}
recompute_num_layers={{recompute_num_layers}}
recompute_granularity={{recompute_granularity}}
recompute_method={{recompute_method}}
empty_unused_memory_level={{empty_unused_memory_level}}
fp={{fp}}

PROJECT_ROOT=/data/work/hiroaki.hosokawa/llm_book
TRAIN_SCRIPT=$PROJECT_ROOT/src/tune/tuning_train.sh
TOKENIZER_PATH=$PROJECT_ROOT/models/Llama-3.1-70B
OUTPUT_DIR=$PROJECT_ROOT/outputs/tuned
HOSTFILE_PATH=/opt/faib/etc/hostfile.txt

mpirun --hostfile $HOSTFILE_PATH \
    -x MASTER_ADDR \
    $TRAIN_SCRIPT \
    $OUTPUT_DIR \
    $TOKENIZER_PATH \
    $gbs \
    $mbs \
    $pp \
    $cp \
    $tp \
    $recompute_num_layers \
    $recompute_granularity \
    $recompute_method \
    $empty_unused_memory_level \
    $fp
