import itertools
from collections import deque
from enum import Enum, auto
from typing import Any, Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from functools import cache

import sys
import os
import time
import logging

logger = logging.getLogger(__name__)

import torch
from torch import nn
import torch.distributed as dist


class AFDForwardStage(Enum):
    AFD_FORWARD_STAGE_A = auto()
    AFD_FORWARD_STAGE_F = auto()


class AFDStageScheduleGenerator:
    Schedule = List[Tuple[AFDForwardStage, int, int]]

    @staticmethod
    def ffn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        for l, m in itertools.product(range(num_layers), range(m_stage)):
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
            schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l, m))
        return schedule

    @staticmethod
    def attn_stage(num_layers: int, m_stage: int) -> Schedule:
        schedule = []
        if num_layers == 1:
            return [
                (AFDForwardStage.AFD_FORWARD_STAGE_A, 0, m) for m in range(m_stage)
            ] + [(AFDForwardStage.AFD_FORWARD_STAGE_F, 0, m) for m in range(m_stage)]
        for l, m in itertools.product(range(num_layers + 1), range(m_stage)):
            if l > 0:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_F, l - 1, m))
            if l < num_layers:
                schedule.append((AFDForwardStage.AFD_FORWARD_STAGE_A, l, m))
        return schedule


class FifoTensorCommunicator(ABC):
    @abstractmethod
    def recv_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def send_tensor(self, x: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def init(self, dp_rank, dp_size, tp_rank, tp_size):
        raise NotImplementedError


class StepMeshTensorCache(object):
    def __init__(self, ten=None, key=0):
        self.push_tensor = ten
        self.pull_tensor = ten

        self.push_key = key
        self.pull_key = key + 1

        self.h = None


def stepmesh_scheduler():
    import setproctitle

    setproctitle.setproctitle("stepmesh_scheduler")

    os.environ["DMLC_ROLE"] = "scheduler"

    logger.info(
        "StepMesh scheduler: DMLC_PS_ROOT_URI=%s" % os.environ["DMLC_NODE_HOST"]
    )
    import fserver
    import fserver.fserver_lib as f

    logger.info("StepMesh scheduler init done.")

    while True:
        time.sleep(10000)

class StepMeshTensorCommunicator(FifoTensorCommunicator):
    def __init__(self):
        super().__init__()

        self.key = 0
        self.comm_ids = []
        self.waits = []
        self.free_tensors = {}
        self.register_buf = {}
        self.buf_size_history = []

    def init(self, dp_rank, dp_size, tp_rank, tp_size):
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        import fserver
        import fserver.fserver_lib as f

        self.start_stepmesh_scheduler()
        if afd_is_attn():
            time.sleep(10)  # wait scheduler
        logger.info("%s init..." % os.environ["DMLC_ROLE"])
        logger.info("%s init done." % os.environ["DMLC_ROLE"])
        self.f = f

    def env_def(self, env, v):
        if os.environ.get(env) == None:
            os.environ[env] = v

    def get_node_ip(self):
        if os.environ.get("DMLC_NODE_HOST") != None:
            return

        import psutil

        interface_name = os.environ.get("DMLC_INTERFACE")

        interfaces = psutil.net_if_addrs()

        if interface_name not in interfaces:
            logger.info("Invalid DMLC_INTERFACE %s" % interface_name)
            return

        for addr in interfaces[interface_name]:
            if addr.family == 2:  # socket.AF_INET
                os.environ["DMLC_NODE_HOST"] = addr.address
                break

    def start_stepmesh_scheduler(self):
        self.get_node_ip()

        self.env_def("DMLC_NODE_RANK", str(self.dp_rank if self.dp_rank else 0))
        self.env_def("DMLC_NUM_SERVER", "1")
        self.env_def("DMLC_NUM_WORKER", "1")
        self.env_def("DMLC_GROUP_SIZE", "1")
        self.env_def("DMLC_PS_ROOT_PORT", "8123")
        self.env_def("DMLC_ENABLE_RDMA", "ibverbs")
        self.env_def("STEPMESH_GPU", str(torch.cuda.current_device()))
        self.env_def("DMLC_INSTANCE_ID", str(self.tp_rank if self.tp_rank else 0))

        if afd_is_attn():
            os.environ["DMLC_ROLE"] = "worker"
        else:
            os.environ["DMLC_ROLE"] = "server"

        if os.environ["DMLC_ROLE"] != "worker":
            return

        if os.environ.get("DMLC_NODE_RANK") != "0":
            return

        if os.environ["STEPMESH_GPU"] != "0":
            return

        if os.environ.get("STEPMESH_SCHEDULER_STARTED") == "1":
            return

        os.environ["STEPMESH_SCHEDULER_STARTED"] = "1"
        os.environ["DMLC_NODE_HOST"] = os.environ["DMLC_PS_ROOT_URI"]

        import multiprocessing

        p = multiprocessing.Process(target=stepmesh_scheduler)
        p.daemon = True
        p.start()

    def attn_send(self, x):
        self.attn_communicator.send(x)

    def attn_recv(self):
        pull_tensor = self.attn_communicator.recv()
        return pull_tensor

    def ffn_send(self, x):
        if self.num_worker_per_server == 1:
            x = self.ffn_communicator.send(x)
        else:
            lengths = self.comm_ids.pop(0)
            x = torch.split(x, lengths, dim=0)
            self.ffn_communicator.send_list(x)

    def ffn_recv(self):
        if self.num_worker_per_server == 1:
            x = [self.ffn_communicator.recv()]
        else:
            x = self.ffn_communicator.recv_list()

        # attention send x.shape[1] when x.shape[1] is 0
        lengths = []
        for i in range(len(x)):
            lengths.append(x[i].size(0))
        self.comm_ids.append(lengths)
        return torch.cat(x, dim=0)

    def recv_tensor(self) -> torch.Tensor:
        if afd_is_attn():
            return self.attn_recv()
        else:
            return self.ffn_recv()

    def send_tensor(self, x: torch.Tensor):
        if afd_is_attn():
            self.attn_send(x)
        else:
            self.ffn_send(x)


@cache
def get_tensor_communicator() -> FifoTensorCommunicator:
    if os.environ.get("DMLC_INTERFACE"):
        sc = StepMeshTensorCommunicator()
        sc.init(0, 0, 0, 0)
        return sc
    else:
        assert False


IS_ATTENTION = os.environ.get("ROLE") == "ATTN"


def get_afd_mirco_batch() -> int:
    return 3


def afd_is_ffn():
    return not IS_ATTENTION


def afd_is_attn():
    return IS_ATTENTION


def model_forward_afd(
    layers,
    a,
    b,
    c,
):
    num_layers = len(layers)
    m_stage = get_afd_mirco_batch()

    stage_outputs: Dict[AFDForwardStage, deque[dict[Any, Any]]] = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: deque(),
        AFDForwardStage.AFD_FORWARD_STAGE_F: deque(),
    }

    stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].extend(
        [
            dict(
                hidden_states=a,
            ),
            dict(
                hidden_states=b,
            ),
            dict(
                hidden_states=c,
            ),
        ]
    )

    def forward_A(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
        print(f"forward_A=================: {mirco_batch_idx}")
        hidden_states = layers[layer_id].forward_afd_A(
            # inputs_args[mirco_batch_idx],
            inputs_args["hidden_states"],
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].append(
            dict(
                hidden_states=hidden_states,
            )
        )

    def forward_F(layer_id: int, mirco_batch_idx: int):
        inputs_args = stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_A].popleft()
        print(f"forward_F=================: {mirco_batch_idx}")
        hidden_states = layers[layer_id].forward_afd_F(
            # inputs_args[mirco_batch_idx],
            inputs_args["hidden_states"],
        )
        stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].append(
            dict(
                hidden_states=hidden_states,
            )
        )

    stage_executors = {
        AFDForwardStage.AFD_FORWARD_STAGE_A: forward_A,
        AFDForwardStage.AFD_FORWARD_STAGE_F: forward_F,
    }

    pipeline_stages = None
    if afd_is_attn():
        pipeline_stages = AFDStageScheduleGenerator.attn_stage(num_layers, m_stage)
    elif afd_is_ffn():
        pipeline_stages = AFDStageScheduleGenerator.ffn_stage(num_layers, m_stage)
    else:
        raise NotImplementedError()
    for type, *args_ in pipeline_stages:
        print(f"====================== {args_}")
        stage_executors.get(type)(*args_)

    try:
        results = [
            stage_outputs[AFDForwardStage.AFD_FORWARD_STAGE_F].popleft()
            for _ in range(m_stage)
        ]
    except IndexError:
        raise ValueError(
            "model_forward_afd: impossible path, a potential implementation bug?"
        )

    # all_hidden_states = zip(
    #     *((res["hidden_states"]) for res in results)
    # )

    if afd_is_attn():
        all_hidden_states = [res["hidden_states"] for res in results]

        return torch.cat(all_hidden_states, dim=0)


class AFDCommunicator:
    def __init__(self):
        self.empty_tensor = torch.empty(0, dtype=torch.float32, device="cuda:0")

    @abstractmethod
    def attn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def ffn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AFDCommunicatorATTN(AFDCommunicator):
    def __init__(self):
        super().__init__()

    def attn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        get_tensor_communicator().send_tensor(hidden_states)
        return self.empty_tensor

    def ffn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return get_tensor_communicator().recv_tensor()


class AFDCommunicatorFFN(AFDCommunicator):
    def __init__(self):
        super().__init__()

    def attn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return get_tensor_communicator().recv_tensor()

    def ffn_transmit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        get_tensor_communicator().send_tensor(hidden_states)
        return self.empty_tensor


NUM_LAYERS = 1


class TestModule:
    def __call__(self, *args, **kvargs):
        return self.forward(*args, **kvargs)


class ATTN(TestModule):
    def __init__(self):
        pass

    def forward(self, hs):
        return hs


class FFN(TestModule):
    def __init__(self):
        pass

    def forward(self, hs):
        return hs


class DecodeLayer(TestModule):
    def __init__(self):
        self.attn = ATTN()
        self.ffn = FFN()
        if afd_is_attn():
            self.afd_communicator = AFDCommunicatorATTN()
        elif afd_is_ffn():
            self.afd_communicator = AFDCommunicatorFFN()

    def forward(self, hs):
        hs = self.attn(hs)
        return self.ffn(hs)

    def forward_afd_A(self, hs):
        hs = self.attn(hs)
        hs = self.afd_communicator.attn_transmit(hs)
        return hs

    def forward_afd_F(self, hs):
        hs = self.ffn(hs)
        hs = self.afd_communicator.ffn_transmit(hs)
        return hs


class TestModel(TestModule):
    def __init__(self):
        self.layers = [DecodeLayer() for it in range(NUM_LAYERS)]

    def forward(self, hs):
        for layer in self.layers:
            hs = self.layer(hs)
        return hs


def main():
    model = TestModel()
    # model()

    results = []

    num_step = 2

    for i in range(num_step):
        print("step one======================================================")
        idx = i * 3
        a = torch.empty([3 + idx + 0, 4], dtype=torch.float32, device="cuda:0")
        b = torch.empty([3 + idx + 1, 4], dtype=torch.float32, device="cuda:0")
        c = torch.empty([3 + idx + 2, 4], dtype=torch.float32, device="cuda:0")
        res = model_forward_afd(model.layers, a, b, c)
        results.append(res)

        if len(results) > 10:
            results = results[1:]

    print("step end")

    if afd_is_attn():
        pass


if __name__ == "__main__":
    main()
