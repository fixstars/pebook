#!/bin/bash -x

# arguments
PROJECT_ROOT=/data/work/hiroaki.hosokawa/llm_book
N_TRIALS=30

python $PROJECT_ROOT/src/tune/tune_megatron.py --n-trials $N_TRIALS
