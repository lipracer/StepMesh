THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
function cleanup() {
    echo "kill all testing process of ps lite for user $USER"
    # pkill -9 -f test_bench
    pkill -9 -f test_pushpull_cache
    sleep 1
}
trap cleanup EXIT
# cleanup

# common setup
export BIN=${BIN:-test_n_a_n_f}
# export DMLC_INTERFACE=${RNIC:-brainpf_bond0}
export SCHEDULER_IP=$(ip -o -4 addr | grep ${RNIC} | awk '{print $4}' | cut -d'/' -f1)
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_GROUP_SIZE=2
export DMLC_PS_ROOT_URI=$SCHEDULER_IP  # scheduler's RDMA interface IP 
export DMLC_PS_ROOT_PORT=8123     # scheduler's port (can random choose)
export DMLC_ENABLE_RDMA=ibverbs
export DMLC_INTERFACE=auto
# export STEPMESH_BIND_CPU_CORE=1

export DMLC_NODE_HOST=${SCHEDULER_IP}
export DMLC_INTERFACE=auto
export STEPMESH_SPLIT_QP_LAG=0
export STEPMESH_GPU=0
# export PS_VERBOSE=1

DMLC_ROLE=scheduler python3 $THIS_DIR/$BIN.py 2>&1 | tee scheduler.log &
sleep 1

DMLC_NODE_RANK=0 CUDA_VISIBLE_DEVICES=0,1,2,3 STEPMESH_GPU=0 DMLC_ROLE=worker python3 $THIS_DIR/$BIN.py $@ 2>&1 | tee worker0.log &
DMLC_NODE_RANK=0 CUDA_VISIBLE_DEVICES=0,1,2,3 STEPMESH_GPU=1 DMLC_ROLE=worker python3 $THIS_DIR/$BIN.py $@ 2>&1 | tee worker1.log &

DMLC_NODE_RANK=0 CUDA_VISIBLE_DEVICES=4,5,6,7 STEPMESH_GPU=0 DMLC_ROLE=server python3 $THIS_DIR/$BIN.py $@ 2>&1 | tee server0.log &
DMLC_NODE_RANK=0 CUDA_VISIBLE_DEVICES=4,5,6,7 STEPMESH_GPU=1 DMLC_ROLE=server python3 $THIS_DIR/$BIN.py $@ 2>&1 | tee server1.log

wait
