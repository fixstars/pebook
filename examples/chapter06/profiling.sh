#!/bin/bash

unset KINETO_USE_DAEMON
unset KINETO_IPC_SOCKET_DIR
unset KINETO_DAEMON_INIT_DELAY_S

# nsys baseline
/opt/openmpi/bin/mpirun -x MASTER_ADDR=192.168.1.2 --host 192.168.1.1:120,192.168.1.2:120 --map-by ppr:8:node --cpus-per-rank 15 --mca plm_rsh_args '-o StrictHostKeyChecking=no -p 53412' \
    ../train_70b_gbs128_adam_nsys.sh \
    ./checkpoints/Llama-3.1-70B-mega \
    ./checkpoints_save/Llama-3.1-70B-mega \
    /data/work/hiroaki.hosokawa/llm_book/models/Llama-3.1-70B \
    ./datasets > baseline_nsys.log 2>&1

# torchprof baseline
/opt/openmpi/bin/mpirun -x MASTER_ADDR=192.168.1.2 --host 192.168.1.1:120,192.168.1.2:120 --map-by ppr:8:node --cpus-per-rank 15 --mca plm_rsh_args '-o StrictHostKeyChecking=no -p 53412' \
    ../train_70b_gbs128_adam_torchprof.sh \
    ./checkpoints/Llama-3.1-70B-mega \
    ./checkpoints_save/Llama-3.1-70B-mega \
    /data/work/hiroaki.hosokawa/llm_book/models/Llama-3.1-70B \
    ./datasets > baseline_torchprof.log 2>&1

# nsys tuned
# /opt/openmpi/bin/mpirun -x MASTER_ADDR=192.168.1.2 --host 192.168.1.1:120,192.168.1.2:120 --map-by ppr:8:node --cpus-per-rank 15 --mca plm_rsh_args '-o StrictHostKeyChecking=no -p 53412' \
#     ../train_70b_gbs128_adam_tuned_nsys.sh \
#     ./checkpoints_tuned/Llama-3.1-70B-mega \
#     ./checkpoints_save/Llama-3.1-70B-mega \
#     /data/work/hiroaki.hosokawa/llm_book/models/Llama-3.1-70B \
#     ./datasets > tuned_nsys.log 2>&1

# # torchprof tuned
# /opt/openmpi/bin/mpirun -x MASTER_ADDR=192.168.1.2 --host 192.168.1.1:120,192.168.1.2:120 --map-by ppr:8:node --cpus-per-rank 15 --mca plm_rsh_args '-o StrictHostKeyChecking=no -p 53412' \
#     ../train_70b_gbs128_adam_tuned_torchprof.sh \
#     ./checkpoints_tuned/Llama-3.1-70B-mega \
#     ./checkpoints_save/Llama-3.1-70B-mega \
#     /data/work/hiroaki.hosokawa/llm_book/models/Llama-3.1-70B \
#     ./datasets > tuned_torchprof.log 2>&1
