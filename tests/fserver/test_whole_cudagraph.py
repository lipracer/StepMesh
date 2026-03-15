import torch, os, sys
import time
from contextlib import nullcontext

import fserver

f = fserver.fslib()

num_layers = 2
num_micro_batch = 3
request = [
    2,
]

# request = request * 1000

is_worker = os.environ.get("DMLC_ROLE") == "worker"
is_server = os.environ.get("DMLC_ROLE") == "server"

gpu = os.environ.get("STEPMESH_GPU")

def attn_op(t):
    return t + 1

def ffn_op(t):
    return t + 2

if is_worker:
    torch.set_default_device("cuda:{}".format(gpu))

    attn_communicator = fserver.AfdTensorCommunicatorATTN(
        f,
        num_micro_batch,
        torch.empty(
            [num_micro_batch, 1024 * 1024 * 4], dtype=torch.int8, device=f"cuda:{gpu}"
        ),
        enable_cg=True,
    )

    inputs = []
    def attn(lid=0):
        global inputs
        inputs[0] = attn_op(inputs[0])
        if lid < num_layers:
            attn_communicator.send(inputs[0])
            inputs = inputs[1:]

    def ffn(lid=0):
        global inputs
        if lid <= num_layers:
            inputs.append(attn_communicator.recv())

    pipeline = [attn]
    pipeline *= num_micro_batch
    one_layer = [ffn, attn]
    for _ in range(num_layers):
        for m in range(num_micro_batch):
            pipeline += one_layer

    pipeline.extend([ffn] * num_micro_batch)

    # A A A (F A) F A F A F A F A F A F F F

    i0 = torch.ones(
        [num_micro_batch, 4], dtype=torch.float32, device=f"cuda:{gpu}", requires_grad=True
    )

    graphs = []

    for req in request:
        graphs.append(torch.cuda.CUDAGraph())
        i0 = torch.ones(
            [req, 4], dtype=torch.float32, device=f"cuda:{gpu}"
        )
        i1 = i0 + 1
        i2 = i0 + 2
        inputs = [i0, i1, i2]

        fserver.capture_begin(req, num_layers, num_micro_batch)
        with torch.cuda.graph(graphs[-1]):
        # with nullcontext():
            layer_id = 0
            for idx, pp in enumerate(pipeline):
                layer_id = 0 if idx < num_micro_batch else (idx - num_micro_batch) // num_layers // num_micro_batch + 1
                pp(layer_id)

        fserver.capture_end()
    
    torch.cuda.synchronize()
    print("worker capture done")
    f.cudagraph_replay(2)
    graphs[-1].replay()

    # time.sleep(10)

elif is_server:
    torch.set_default_device("cuda:{}".format(gpu))
    ffn_communicator = fserver.AfdTensorCommunicatorFFN(
        f,
        num_micro_batch,
        torch.empty(
            [num_micro_batch, 1024 * 1024 * 4], dtype=torch.int8, device=f"cuda:{gpu}"
        ),
        enable_cg=True,
    )

    pipeline = []
    inputs = []

    def attn():
        global inputs
        inputs.append(ffn_communicator.recv())
    def ffn():
        global inputs
        inputs[0] = ffn_op(inputs[0])
        ffn_communicator.send(inputs[0])
        inputs = inputs[1:]

    pipeline = []
    for _ in range(num_layers):
        for m in range(num_micro_batch):
            pipeline.append(attn)
            pipeline.append(ffn)

    graphs = []

    for req in request:
        graphs.append(torch.cuda.CUDAGraph())
        fserver.capture_begin(req, num_layers, num_micro_batch)
        with torch.cuda.graph(graphs[-1]):
        # with nullcontext():
            for pp in pipeline:
                pp()
        fserver.capture_end()

    torch.cuda.synchronize()
    print("server capture done")
    f.cudagraph_replay(2)
    graphs[-1].replay()

    # print(f"flag:{ffn_communicator.flag_tensor}")
    # time.sleep(10)
    # print(f"flag:{ffn_communicator.flag_tensor}")


f.stop()
