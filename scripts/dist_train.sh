#!/bin/bash
# PointRWKV distributed training script
# Usage: bash scripts/dist_train.sh <NUM_GPUS> <SCRIPT> [OTHER_ARGS]
# Example: bash scripts/dist_train.sh 4 main_cls.py --config cfgs/cls_modelnet40.yaml --exp_name cls_mn40

NUM_GPUS=$1
shift
SCRIPT=$1
shift

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=$((RANDOM + 10000)) \
    ${SCRIPT} \
    --launcher pytorch \
    "$@"
