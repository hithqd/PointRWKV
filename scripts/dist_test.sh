#!/bin/bash
# PointRWKV distributed testing script
# Usage: bash scripts/dist_test.sh <NUM_GPUS> <SCRIPT> [OTHER_ARGS]
# Example: bash scripts/dist_test.sh 1 main_cls.py --config cfgs/cls_modelnet40.yaml --test --ckpt /path/to/ckpt --vote

NUM_GPUS=$1
shift
SCRIPT=$1
shift

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=$((RANDOM + 10000)) \
    ${SCRIPT} \
    --launcher pytorch \
    --test \
    "$@"
