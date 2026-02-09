import torch, os, sys
import time

import fserver

f = fserver.fslib()

num_layers = 2
num_micro_batch = 3
request = [
    2,
    8,
    9,
]

request = request * 1000

is_worker = os.environ.get("DMLC_ROLE") == "worker"
is_server = os.environ.get("DMLC_ROLE") == "server"

gpu = os.environ.get("STEPMESH_GPU")

if is_worker:
    torch.set_default_device("cuda:{}".format(gpu))

    attn_communicator = fserver.AfdTensorCommunicatorATTN(
        f,
        num_micro_batch,
        torch.empty(
            [num_micro_batch, 1024 * 1024 * 4], dtype=torch.int8, device=f"cuda:{gpu}"
        ),
    )

    for req in request:
        for layer in range(num_layers):
            for micro_batch in range(num_micro_batch):
                push_tensors = torch.rand(
                    [req + micro_batch, 8192], dtype=torch.float32, device=f"cuda:{gpu}"
                )
                push_tensors += 1
                print(f"attn send micro_batch:{micro_batch}")
                attn_communicator.send(push_tensors)
                print(f"attn recv micro_batch:{micro_batch}")
                pull_tensors = attn_communicator.recv()
                golden = push_tensors + 3
                torch.equal(pull_tensors, golden)

    print("worker test done")

elif is_server:
    torch.set_default_device("cuda:{}".format(gpu))
    ffn_communicator = fserver.AfdTensorCommunicatorFFN(
        f,
        num_micro_batch,
        torch.empty(
            [num_micro_batch, 1024 * 1024 * 4], dtype=torch.int8, device=f"cuda:{gpu}"
        ),
    )

    for req in request:
        for layer in range(num_layers):
            for micro_batch in range(num_micro_batch):
                hs = ffn_communicator.recv()
                hs += 2
                hs = ffn_communicator.send(hs)

f.stop()
