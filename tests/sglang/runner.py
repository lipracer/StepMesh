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


class StepMeshTensorCommunicatorBase(FifoTensorCommunicator):
    def __init__(self):
        super().__init__()

        self.key = 0
        self.comm_ids = []
        self.waits = []
        self.free_tensors = {}
        self.register_buf = {}
        self.buf_size_history = []

        self.pp_tensors = []

        self.push_tensor = None
        self.pull_tensor = None

        self.cache_tensors = None
        self.recv_tensors = None
        self.respond_tensors = None

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
        if x.size(0) == 0:
            x = torch.empty([1], dtype=x.dtype, device=x.device)

        if self.cache_tensors is None:
            self.cache_tensors = [
                torch.empty([1024, 1024], dtype=x.dtype, device=x.device)
                for i in range(get_afd_mirco_batch())
            ]

            for t in self.cache_tensors:
                self.f.regist_push_pull_buffer(t, [0], t, [0])

        self.key += 2
        if self.key == 8:
            self.key = 2

        mirco_batch_idx = self.key // 2 - 1
        print(f"mirco_batch_idx:{mirco_batch_idx}")
        push_tensor = (
            self.cache_tensors[mirco_batch_idx].view(-1)[: x.numel()].view_as(x)
        )

        pull_tensor = torch.empty_like(x)
        push_tensor.copy_(x)

        self.pp_tensors.append((push_tensor, pull_tensor))

        h = self.f.push_pull([push_tensor], [self.key], [pull_tensor], [self.key + 1])

        self.waits.append(h)

    def attn_recv(self):
        h = self.waits.pop(0)
        self.f.wait(h, timeout_ms=100000)

        push_tensor, pull_tensor = self.pp_tensors.pop(0)

        if pull_tensor.ndim == 1:
            return torch.empty(0, dtype=pull_tensor.dtype, device=pull_tensor.device)

        return pull_tensor

    def ffn_send(self, x):
        # if self.respond_tensors is None:
        #     self.respond_tensors = torch.empty([1024, 1024], dtype=x.dtype, device=x.device)
        #     self.f.register_recv_buffer(self.respond_tensors[0], [0], [2])

        # t = self.respond_tensors[0].view(-1)[:x.numel()].view_as(x)
        # t.copy_(x)
        comm_id = self.comm_ids.pop(0)
        self.f.respond([x], comm_id[0], True)

    def ffn_recv(self):
        if self.recv_tensors is None:
            self.recv_tensors = [
                torch.empty([1024, 1024], dtype=torch.float32, device="cuda:0")
                for i in range(get_afd_mirco_batch())
            ]
            for i in range(len(self.recv_tensors)):
                self.f.register_recv_buffer(self.recv_tensors[i], [0], [2 + i * 2 + 1])

        batches = self.f.get_batch()

        ## batches [
        #     [comm_id, push_tensor_list, key_list],
        #     [comm_id, push_tensor_list, key_list],
        # ]
        # assert len(batches) == 1, "just handle for one worker"

        print(f"get batch========================== {batches}")
        self.comm_ids.append(batches[0])
        return batches[0][1][0]

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
        if x.size(0) == 0:
            x = torch.empty([1], dtype=x.dtype, device=x.device)

        free = self.free_tensors.get(x.shape, [])
        self.free_tensors[x.shape] = free

        if free:
            t = free.pop()
            t.push_tensor.copy_(x)
        else:
            self.key += 2
            t = StepMeshTensorCache(x, self.key)

            if len(self.buf_size_history) > 3:
                oldest_size = self.buf_size_history.pop(0)
                _tensors = self.free_tensors.get(oldest_size)
                if len(_tensors) > 1:
                    del _tensors[-1]
                else:
                    del self.free_tensors[oldest_size]
            self.buf_size_history.append(x.shape)

        t.h = self.f.push_pull(
            [t.push_tensor], [t.push_key], [t.pull_tensor], [t.pull_key]
        )

        self.waits.append(t)

    def attn_recv(self):
        t = self.waits.pop(0)
        self.f.wait(t.h, timeout_ms=100000)

        free = self.free_tensors.get(t.push_tensor.shape, [])
        self.free_tensors[t.push_tensor.shape] = free
        free.append(t)

        if t.pull_tensor.ndim == 1:
            return torch.empty(
                0, dtype=t.pull_tensor.dtype, device=t.pull_tensor.device
            )

        return t.pull_tensor

    def ffn_send(self, x):
        free = self.free_tensors.get(x.shape, [])
        self.free_tensors[x.shape] = free

        if free:
            t = free.pop()
            t.copy_(x)
        else:
            t = torch.empty_like(x)
            t.copy_(x)

            if len(self.buf_size_history) > 3:
                oldest_size = self.buf_size_history.pop(0)
                _tensors = self.free_tensors.get(oldest_size)
                if len(_tensors) > 1:
                    del _tensors[-1]
                else:
                    del self.free_tensors[oldest_size]
            self.buf_size_history.append(x.shape)

        c = self.comm_ids.pop(0)
        for _id, comm_id in enumerate(c[0]):
            _need_event = _id == 0
            idx_range = c[1][_id]
            _res_tensor = t[idx_range[0] : idx_range[1]]
            if _res_tensor.size(0) == 0:
                _res_tensor = torch.empty([1], dtype=_res_tensor.dtype, device=x.device)
            self.f.respond([_res_tensor], comm_id, _need_event)

        free.append(t)

    def ffn_recv(self):
        batches = self.f.get_batch()

        ## batches [
        #     [comm_id, push_tensor_list, key_list],
        #     [comm_id, push_tensor_list, key_list],
        # ]
        # assert len(batches) == 1, "just handle for one worker"

        _comm_ids = []
        _idx_ranges = []
        _tensor_list = []
        _begin_idx = 0

        for batch in batches:
            if batch[1][0].ndim == 1:
                _length = 0
            else:
                _length = batch[1][0].size(0)
                _tensor_list.append(batch[1][0])
            _end_idx = _begin_idx + _length
            _idx_ranges.append([_begin_idx, _end_idx])
            _comm_ids.append(batch[0])
            _begin_idx = _end_idx
        self.comm_ids.append([_comm_ids, _idx_ranges])

        print(f"self.comm_ids:{self.comm_ids}")

        if len(_tensor_list) == 0:
            return torch.empty(
                0, dtype=batches[0][1][0].dtype, device=batches[0][1][0].device
            )
        elif len(_tensor_list) == 1:
            return _tensor_list[0]
        else:
            assert False, "unreachable"
            return torch.cat(_tensor_list, dim=0)

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
        sc = StepMeshTensorCommunicatorBase()
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
