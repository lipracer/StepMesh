import os, sys
import time
import torch
from pathlib import Path
import itertools
import logging
from collections import deque

logger = logging.getLogger(__name__)

logger_verbose = os.environ.get("PS_VERBOSE")
ps_logger = logger.info

if logger_verbose and int(logger_verbose) == 1:
    ps_logger = logger.warning


def my_logger(msg):
    print(msg, flush=True)


# ps_logger = ps_logger

_fserver_lib = None


def fslib():
    old_flags = sys.getdlopenflags()
    # export sysmbol for plugin
    sys.setdlopenflags(sys.getdlopenflags() | 0x100)
    from . import fserver_lib

    sys.setdlopenflags(old_flags)

    plugin_path = os.path.join(Path(__file__).resolve().parent, "libklx_backend.so")
    fserver_lib.init(f"{plugin_path}")
    global _fserver_lib
    _fserver_lib = fserver_lib
    return fserver_lib


class CG_Context:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def __enter__(self):
        # self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.end()
        pass


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
            ps_logger(
                f"key_start:{key_start} {int(torch.cuda.current_device())} {int(torch.cuda.current_device()) << 16}"
            )
            self.keys = [key_start + i for i in range(micro_batch_size)]

        self.flag_key = []
        ps_logger(f"keys:{self.keys}")

        if enable_cache:
            self.regist_cache()

        # NB: don't reuse, send and recv not paired
        self.send_index_gen = itertools.cycle([i for i in range(micro_batch_size)])
        self.recv_index_gen = itertools.cycle([i for i in range(micro_batch_size)])

        self.is_capturing = False
        self.flag_value = [1 for i in range(self.micro_batch_size)]

    def send(self, x):
        micro_batch_index = next(self.send_index_gen)
        if self.is_worker():
            # self.init_cg_if_capturing()
            ps_logger(f"send is_capturing:{self.is_capturing}")

        ps_logger(
            f"send is_capturing:{self.is_capturing} micro_batch_index:{micro_batch_index}"
        )
        assert isinstance(x, torch.Tensor)
        if x.numel() * x.element_size() > self.cache[0].numel():
            self.update_cache(x.numel() * x.element_size())
        if not self.is_capturing:
            self.send_impl(x, micro_batch_index)
        else:
            self.send_impl(x, micro_batch_index)
            self.write_remote_flag(micro_batch_index)

    def send_list(self, x):
        assert False
        assert isinstance(x, list)
        self.send_list_impl(x, next(self.send_index_gen))

    def recv(self):
        micro_batch_index = next(self.recv_index_gen)
        if self.is_server():
            # self.init_cg_if_capturing()
            ps_logger(f"send is_capturing:{self.is_capturing}")

        ps_logger(
            f"recv is_capturing:{self.is_capturing} micro_batch_index:{micro_batch_index}"
        )
        if not self.is_capturing:
            return self.recv_impl(micro_batch_index)
        else:
            res = self.recv_impl(micro_batch_index)
            self.read_remote_flag(micro_batch_index)
            return res

    def recv_list(self):
        micro_batch_index = next(self.recv_index_gen)
        return self.recv_list_impl(micro_batch_index)

    def update_cache(self, new_size):
        assert False, "not implemented please update cache size to {new_size}"

    def is_worker(self):
        return isinstance(self, AfdTensorCommunicatorATTN)

    def is_server(self):
        return isinstance(self, AfdTensorCommunicatorFFN)

    def flag_handshake(self):
        pass

    @property
    def need_event(self):
        return True
        # return not self.is_capturing

    def prepare_cg(self):
        import klxops

        key_start = 1 + int(torch.cuda.current_device()) << 16
        self.flag_key = [
            key_start + i
            for i in range(self.micro_batch_size, 2 * self.micro_batch_size)
        ]
        # cuda graph dummy run
        self.flag_tensor = [
            torch.ones([1], dtype=torch.int32, device="cuda")
            for i in range(self.micro_batch_size)
        ]
        # torch.cuda.synchronize()
        self.regist_flag()

    def init_cg_if_capturing(self, force_capturing=False):
        if force_capturing:
            self.is_capturing = True
        else:
            self.is_capturing = torch.cuda.is_current_stream_capturing()
        if len(self.flag_key) != 0:
            return
        self.prepare_cg()

    def clear_cache(self):
        for c in self.cache:
            c.zero_()

    @property
    def timeout(self):
        return 100000


class AfdTensorCommunicatorATTN(AfdTensorCommunicator):
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        self.handlers = deque()
        self.flag_handlers = deque()

    def regist_cache(self):
        for i in range(self.micro_batch_size):
            self.f.regist_push_pull_buffer(self.cache[i], [0], self.cache[i], [0])

    def regist_flag(self):
        for i in range(self.micro_batch_size):
            self.f.regist_push_pull_buffer(
                self.flag_tensor[i], [0], self.flag_tensor[i], [0]
            )
        self.f.cudagraph_regist_flag(self.flag_tensor)

    def send_impl(self, x, micro_batch_index):
        if self.is_capturing:
            self.f.cudagraph_set_stage(micro_batch_index)
        ps_logger(
            f"attn micro_batch:{micro_batch_index} send x:{x.shape} key:{self.keys[micro_batch_index]}"
        )
        send_buffer = self.cache[micro_batch_index]
        send_buffer = send_buffer.view(x.dtype)
        send_buffer = send_buffer.view(-1)[: x.numel()].view_as(x)
        send_buffer.copy_(x)

        h = self.f.push_pull(
            [send_buffer],
            [self.keys[micro_batch_index]],
            [send_buffer],
            [self.keys[micro_batch_index]],
            need_event=self.need_event,
        )
        ps_logger(
            f"[WORKER] send_impl: micro_batch={micro_batch_index} push_pull returned handler={h} is_capturing={self.is_capturing}"
        )
        self.handlers.append((h, send_buffer))

    def recv_impl(self, micro_batch_index):
        ps_logger(f"attn micro_batch:{micro_batch_index} recv")
        handler, res = self.handlers.popleft()
        ps_logger(
            f"[WORKER] recv_impl: micro_batch={micro_batch_index} waiting handler={handler}"
        )
        self.f.wait(handler, self.timeout)
        ps_logger(
            f"[WORKER] recv_impl: micro_batch={micro_batch_index} handler={handler} done"
        )
        return res

    def write_remote_flag(self, micro_batch_index):
        torch.ops.klxops.write_flag(
            self.flag_tensor[micro_batch_index], self.flag_value[micro_batch_index]
        )
        self.flag_value[micro_batch_index] += 1
        # need not event, because write flag has already been waited by send data
        h = self.f.push_pull(
            [self.flag_tensor[micro_batch_index]],
            [self.flag_key[micro_batch_index]],
            [self.flag_tensor[micro_batch_index]],
            [self.flag_key[micro_batch_index]],
            need_event=False,
        )
        self.flag_handlers.append(h)
        ps_logger(
            f"write_remote_flag push tensor:{self.flag_tensor[micro_batch_index]}"
        )

    def read_remote_flag(self, micro_batch_index):
        ps_logger(f"attn read_remote_flag before wait")
        handler = self.flag_handlers.popleft()
        self.f.wait(handler, self.timeout)
        ps_logger(f"attn read_remote_flag after wait")
        torch.ops.klxops.wait_flag(
            self.flag_tensor[micro_batch_index], self.flag_value[micro_batch_index]
        )
        self.flag_value[micro_batch_index] += 1


class AfdTensorCommunicatorFFN(AfdTensorCommunicator):
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        self.comm_ids = deque()
        self.flag_q = deque()

    def regist_cache(self):
        num_server = os.environ.get("DMLC_NUM_SERVER")
        num_worker = os.environ.get("DMLC_NUM_WORKER")

        num_server = int(num_server)
        num_worker = int(num_worker)

        if num_server == num_worker:
            # d0 is micro batch index
            # d1 is cahce size
            for micro_batch in range(self.micro_batch_size):
                self.f.register_recv_buffer(
                    self.cache[micro_batch], [0], [self.keys[micro_batch]]
                )
        else:
            assert num_worker >= num_server
            num_a_per_f = num_worker // num_server
            # d0 is rank
            # d1 is micro batch index
            # d2 is cahce size
            for rank in range(num_a_per_f):
                for micro_batch in range(self.micro_batch_size):
                    self.f.register_recv_buffer(
                        self.cache[rank][micro_batch], [rank], [self.keys[micro_batch]]
                    )

    def regist_flag(self):
        num_server = os.environ.get("DMLC_NUM_SERVER")
        num_worker = os.environ.get("DMLC_NUM_WORKER")

        num_server = int(num_server)
        num_worker = int(num_worker)

        if num_server == num_worker:
            for micro_batch in range(self.micro_batch_size):
                self.f.register_recv_buffer(
                    self.flag_tensor[micro_batch], [0], [self.flag_key[micro_batch]]
                )
        else:
            assert num_worker >= num_server
            num_a_per_f = num_worker // num_server
            for rank in range(num_a_per_f):
                self.f.register_recv_buffer(
                    self.flag_tensor[rank][micro_batch],
                    [rank],
                    [self.keys[micro_batch]],
                )

        self.f.cudagraph_regist_flag(self.flag_tensor)

    def send_impl(self, x, micro_batch_index):
        ps_logger(f"ffn micro_batch:{micro_batch_index} send x:{x.shape}")
        comm_id = self.comm_ids.popleft()
        ps_logger(
            f"[SERVER] send_impl: micro_batch={micro_batch_index} responding with handler={comm_id[0]} is_capturing={self.is_capturing}"
        )
        self.f.respond([x], comm_id[0], need_event=self.need_event)

    def send_list_impl(self, x, micro_batch_index):
        comm_id = self.comm_ids.popleft()
        for i in range(len(comm_id)):
            self.f.respond([x[i]], comm_id[i], need_event=self.need_event)

    def recv_impl(self, micro_batch_index):
        ps_logger(f"ffn micro_batch:{micro_batch_index} recv")
        if self.is_capturing:
            self.f.cudagraph_set_stage(micro_batch_index)
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
        ps_logger(
            f"[SERVER] recv_impl: micro_batch={micro_batch_index} got handlers={comm_ids} is_capturing={self.is_capturing}"
        )
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

    def write_remote_flag(self, micro_batch_index):
        torch.ops.klxops.write_flag(
            self.flag_tensor[micro_batch_index], self.flag_value[micro_batch_index]
        )
        batchs = self.flag_q.popleft()
        self.f.respond(batchs[0][1], batchs[0][0], need_event=False)
        self.flag_value[micro_batch_index] += 1

    def read_remote_flag(self, micro_batch_index):
        ps_logger(f"ffn micro_batch:{micro_batch_index} read flag")
        batchs = self.f.get_batch()
        self.flag_q.append(batchs)
        torch.ops.klxops.wait_flag(
            self.flag_tensor[micro_batch_index], self.flag_value[micro_batch_index]
        )
        self.flag_value[micro_batch_index] += 1


def cg_set_config(num_micro_batch):
    _fserver_lib.cudagraph_set_config(num_micro_batch)


def capture_begin():
    _fserver_lib.capture_begin()


def capture_end():
    res = _fserver_lib.capture_end()
    return res


class SMCudaGraph(torch.cuda.CUDAGraph):
    def capture_begin(self, *args, **kwargs) -> None:
        capture_begin()
        self.communicator.init_cg_if_capturing(True)
        super().capture_begin(*args, **kwargs)

    def capture_end(self) -> None:
        super().capture_end()
        self.cur_sm_g = capture_end()
        [f.fill_(0) for f in self.communicator.flag_tensor]
        # fix hand then uncommmon
        # _fserver_lib.barrier(True, True)

    def replay(self) -> None:
        _fserver_lib.cudagraph_replay(self.cur_sm_g)
        super().replay()

    def set_communicator(self, c):
        self.communicator = c
