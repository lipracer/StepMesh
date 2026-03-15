import torch, os, sys
import time

import fserver

f = fserver.fslib()

num_layers = 2
num_micro_batch = 3
request = [
    2,
]

is_worker = os.environ.get("DMLC_ROLE") == "worker"
is_server = os.environ.get("DMLC_ROLE") == "server"

gpu = os.environ.get("STEPMESH_GPU")

batch_size = 2

fserver.cg_set_config(num_micro_batch)

cache_tensor = [torch.zeros([num_micro_batch, 2 * 8192 * 4], dtype=torch.int8, device=f"cuda:{gpu}") for i in range(num_micro_batch)]

if is_worker:
    torch.set_default_device("cuda:{}".format(gpu))

    attn_communicator = fserver.AfdTensorCommunicatorATTN(
        f,
        num_micro_batch,
        cache_tensor,
    )


    graphs = []
    hidden_size = 8192
    graphs.append(fserver.SMCudaGraph())
    graphs[-1].set_communicator(attn_communicator)

    p0 = torch.ones(
        [batch_size, hidden_size], dtype=torch.float32, device=f"cuda:{gpu}"
    )
    p1 = torch.ones(
        [batch_size, hidden_size], dtype=torch.float32, device=f"cuda:{gpu}"
    )
    p2 = torch.ones(
        [batch_size, hidden_size], dtype=torch.float32, device=f"cuda:{gpu}"
    )

    
    org_inputs = [p0, p1, p2]
    inputs = [p0, p1, p2]


    def run_forward():
        global push_tensors
        for i in range(num_layers):
            for micro_batch in range(num_micro_batch):
                inputs[micro_batch] = inputs[micro_batch] + (micro_batch + 1)
                attn_communicator.send(inputs[micro_batch])
                inputs[micro_batch] = attn_communicator.recv()


    graphs[-1].enable_debug_mode()
    with torch.cuda.graph(graphs[-1]):
        run_forward()
    torch.cuda.synchronize()
    graphs[-1].debug_dump("worker_graph")

    # torch.cuda.synchronize()
    # print(f"worker test done inputs:{inputs} flag:{attn_communicator.flag_tensor}", flush=True)

    # attn_communicator.clear_cache()
    graphs[-1].replay()
    # torch.cuda.synchronize()
    print(f"replay test=1 done inputs:{inputs}", flush=True)

    assert torch.allclose(inputs[0], org_inputs[0] + 1 + 2 + 1 + 2), f"input:{inputs[0]} vs {org_inputs[0] + 1 + 2 + 1 + 2}"
    assert torch.allclose(inputs[1], org_inputs[1] + 2 + 4 + 2 + 4)
    assert torch.allclose(inputs[2], org_inputs[2] + 3 + 6 + 3 + 6)

    graphs[-1].replay()
    # torch.cuda.synchronize()
    print(f"replay test=2 done push_tensors:{inputs}", flush=True)

    # torch.cuda.synchronize()


    run_forward()
    # torch.cuda.synchronize()
    print(f"new forward test=3 done push_tensors:{inputs}", flush=True)

    run_forward()
    # torch.cuda.synchronize()
    print(f"new forward test=3 done push_tensors:{inputs}", flush=True)

    graphs[-1].replay()
    # torch.cuda.synchronize()
    print(f"replay test=3 done push_tensors:{inputs}", flush=True)


elif is_server:
    torch.set_default_device("cuda:{}".format(gpu))
    ffn_communicator = fserver.AfdTensorCommunicatorFFN(
        f,
        num_micro_batch,
        cache_tensor,
    )
    first_run = [False for i in range(num_micro_batch)]

    def run_forward():
        for i in range(num_layers):
            for micro_batch in range(num_micro_batch):
                hs = ffn_communicator.recv()
                if first_run[micro_batch]:
                    print(f"recv hs:{hs}", flush=True)

                hs += 2 * (micro_batch + 1)

                if first_run[micro_batch]:
                    print(f"send hs:{hs}", flush=True)
                hs = ffn_communicator.send(hs)

                if not first_run[micro_batch] and i == 1:
                    first_run[micro_batch] = True


    graphs = []
    graphs.append(fserver.SMCudaGraph())
    graphs[-1].set_communicator(ffn_communicator)

    graphs[-1].enable_debug_mode()
    with torch.cuda.graph(graphs[-1]):
        run_forward()
    torch.cuda.synchronize()
    print("server test done", flush=True)
    # ffn_communicator.clear_cache()

    graphs[-1].debug_dump("server_graph")


    graphs[-1].replay()
    # torch.cuda.synchronize()
    print("server replay=1 done", flush=True)

    graphs[-1].replay()
    print("server replay=2 done", flush=True)

    # torch.cuda.synchronize()

    # check rdma status
    run_forward()
    print("server new forward done", flush=True)

    run_forward()
    print("server new forward done", flush=True)

    graphs[-1].replay()
    print("server replay=3 done", flush=True)

if "scheduler" == os.environ.get("DMLC_ROLE"):
    time.sleep(10)
print(f"before stop===================================", flush=True)
f.stop()
