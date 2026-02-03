#!/bin/bash

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    # pkill -9 -f test_bench
    pkill -9 -f runner
    pkill -9 -f stepmesh_scheduler
    sleep 1
}
trap cleanup EXIT


export AFD_SCHED_HOST=127.0.0.1
export RNIC=eth2
export DMLC_PS_ROOT_URI=$(ip -o -4 addr | grep ${RNIC} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_PS_ROOT_PORT=8123

export DMLC_NUM_SERVER=1
export DMLC_NUM_WORKER=1
export DMLC_GROUP_SIZE=1


# export DMLC_INTERFACE=auto
export DMLC_INTERFACE=$RNIC
export STEPMESH_SPLIT_QP_LAG=0
# export STEPMESH_BIND_CPU_CORE=1
export STEPMESH_GPU=0
export PS_VERBOSE=1

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export BIN=${BIN:-runner.py}

ROLE=ATTN python3 $THIS_DIR/$BIN 2>&1 | tee attn.log &
ROLE=FFN python3 $THIS_DIR/$BIN 2>&1 | tee ffn.log

wait