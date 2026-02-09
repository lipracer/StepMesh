import os, sys
import time
import torch
from pathlib import Path
import itertools
import logging
from collections import deque

logger = logging.getLogger(__name__)


def fslib():
    old_flags = sys.getdlopenflags()
    # export sysmbol for plugin
    sys.setdlopenflags(sys.getdlopenflags() | 0x100)
    from . import fserver_lib

    sys.setdlopenflags(old_flags)

    plugin_path = os.path.join(Path(__file__).resolve().parent, "libklx_backend.so")
    fserver_lib.init(f"{plugin_path}")
    return fserver_lib


class AfdTensorCommunicator:
    def __init__(
        self,
        fserver,
        micro_batch_size,
        cache,
        keys=None,
        enable_cache=True,
    ):
        self.f = fserver
        self.micro_batch_size = micro_batch_size
        self.cache = cache
        self.keys = keys

        if self.keys is None:
            key_start = 1 + int(torch.cuda.current_device()) << 16
            self.keys = [key_start + i for i in range(micro_batch_size)]

        if enable_cache:
            self.regist_cache()

        # NB: don't reuse, send and recv not paired
        self.send_index_gen = itertools.cycle([i for i in range(micro_batch_size)])
        self.recv_index_gen = itertools.cycle([i for i in range(micro_batch_size)])

    def send(self, x):
        assert isinstance(x, torch.Tensor)
        if x.numel() * x.element_size() > self.cache[0].numel():
            self.update_cache(x.numel() * x.element_size())
        self.send_impl(x, next(self.send_index_gen))

    def send_list(self, x):
        assert isinstance(x, list)
        self.send_list_impl(x, next(self.send_index_gen))

    def recv(self):
        micro_batch_index = next(self.recv_index_gen)
        return self.recv_impl(micro_batch_index)

    def recv_list(self):
        micro_batch_index = next(self.recv_index_gen)
        return self.recv_list_impl(micro_batch_index)

    def update_cache(self, new_size):
        assert False, "not implemented please update cache size to {new_size}"


class AfdTensorCommunicatorATTN(AfdTensorCommunicator):
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        self.handlers = deque()

    def regist_cache(self):
        for i in range(self.micro_batch_size):
            self.f.regist_push_pull_buffer(self.cache[i], [0], self.cache[i], [0])

    def send_impl(self, x, micro_batch_index):
        send_buffer = self.cache[micro_batch_index]
        send_buffer = send_buffer.view(x.dtype)
        send_buffer = send_buffer.view(-1)[: x.numel()].view_as(x)
        send_buffer.copy_(x)

        h = self.f.push_pull(
            [send_buffer],
            [self.keys[micro_batch_index]],
            [send_buffer],
            [self.keys[micro_batch_index]],
        )
        self.handlers.append((h, send_buffer))

    def recv_impl(self, micro_batch_index):
        handler, res = self.handlers.popleft()
        self.f.wait(handler)
        return res


class AfdTensorCommunicatorFFN(AfdTensorCommunicator):
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        self.comm_ids = deque()

    def regist_cache(self):
        num_server = os.environ.get("DMLC_NUM_SERVER")
        num_worker = os.environ.get("DMLC_NUM_WORKER")

        num_server = int(num_server)
        num_worker = int(num_worker)

        if num_server == num_worker:
            # d0 is micro batch index
            # d1 is cahce size
            if self.cache.ndim != 2:
                self.cache = self.cache.reshape(self.micro_batch_size, -1)
            for i in range(self.micro_batch_size):
                self.f.register_recv_buffer(self.cache[i], [0], [self.keys[i]])
        else:
            assert num_worker >= num_server
            num_a_per_f = num_worker // num_server
            # d0 is rank
            # d1 is micro batch index
            # d2 is cahce size
            if self.cache.ndim != 3:
                self.cache = self.cache.reshape(num_a_per_f, self.micro_batch_size, -1)
            for rank in range(num_a_per_f):
                for micro_batch in range(self.micro_batch_size):
                    self.f.register_recv_buffer(
                        self.cache[rank][micro_batch], [rank], [self.keys[micro_batch]]
                    )

    def send_impl(self, x, micro_batch_index):
        comm_id = self.comm_ids.popleft()
        self.f.respond([x], comm_id[0], True)

    def send_list_impl(self, x, micro_batch_index):
        comm_id = self.comm_ids.popleft()
        for i in range(len(comm_id)):
            self.f.respond([x[i]], comm_id[i], True)

    def recv_impl(self, micro_batch_index):
        """
        batch struct:
        [
            (common_id, [tensor_list], [key_list]),
            (common_id, [tensor_list], [key_list]),
            ....
        ]
        """
        batchs = []
        while len(batchs) == 0:
            batchs = self.f.get_batch()
            if len(batchs) != 0:
                break
            time.sleep(1)
        comm_ids = [b[0] for b in batchs]
        self.comm_ids.append(comm_ids)
        return batchs[0][1][0]

    def recv_list_impl(self, micro_batch_index):
        """
        batch struct:
        [
            (common_id, [tensor_list], [key_list]),
            (common_id, [tensor_list], [key_list]),
            ....
        ]
        """
        batchs = []
        while len(batchs) == 0:
            batchs = self.f.get_batch()
            if len(batchs) != 0:
                break
            time.sleep(1)
        comm_ids = [b[0] for b in batchs]
        self.comm_ids.append(comm_ids)
        return [b[1][0] for b in batchs]
