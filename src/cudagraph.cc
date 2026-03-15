#include "ps/cudagraph.h"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ps/af_tensor_app.h"

namespace ps {

void CommandQ::replay() {
  PS_LOG(INFO) << "replay cmd size:" << cmds.size();
  PS_CHECK(cmds.size() / ctx_->num_micro_batch_ * 2 > 1)
      << "one layer not support";

  for (size_t i = 0, replay_stage = 0; i < cmds.size();
       i += 2, replay_stage++) {
    ctx_->set_stage(replay_stage % ctx_->num_micro_batch_);
    reset_cmd(this, i);
    reset_cmd(this, i + 1);
  }
  PS_CHECK_EQ(cmds.size(), invokers.size());

  for (size_t i = 0, replay_stage = 0; i < cmds.size();
       i += 2, replay_stage++) {
    ctx_->set_stage(replay_stage % ctx_->num_micro_batch_);
    ctx_->layer_id_ = i / 6;
    invokers[i](cmds[i]);
    invokers[i + 1](cmds[i + 1]);
  }
}

std::tuple<void*, void*> CommandQ::alloc_event() {
  PS_CHECK_NE(pool_start_ - pool_end_, kPoolSize);
  auto pos = pool_start_++ & (kPoolSize - 1);
  return std::make_tuple(tensor_events_ + pos, native_events_ + pos);
}

void CommandQ::free_event() { pool_end_--; }

CudaGraphContext::~CudaGraphContext() {}

void CudaGraphContext::destroy() {}

CudaGraphContext& CudaGraphContext::instance() {
  static CudaGraphContext instance;
  return instance;
}

void CudaGraphContext::set_config(int64_t num_micro_batch, Node::Role role) {
  num_micro_batch_ = num_micro_batch;
  role_ = role;
  if (role_ == Node::WORKER) {
    expect_flag_value_.resize(num_micro_batch, 1);
  } else if (role_ == Node::SERVER) {
    expect_flag_value_.resize(num_micro_batch, 2);
  }
}

static auto create_event(CommandQ* q, size_t i) {
  auto [te, ne] = q->alloc_event();
  auto e = new (te) TensorEvent(new (ne) Event(
      {.kind = Event::kGraphEventReplaying,
       .data = CudaGraphContext::instance().create_wait_event()}));
  e->Record();
  return e;
}

void CudaGraphContext::capture_begin() {
  PS_LOG(INFO) << "capture begin";
  is_capturing_ = true;
  cur_graph_ = SMGraph{.cmd_q = new CommandQ(this)};

  if (role_ == Node::WORKER) {
    cmd_q()->reset_cmd = [](CommandQ* q, size_t index) {
      auto req = reinterpret_cast<AFTensorRequest*>(q->cmds[index]);
      req->capture_info.stage = CudaGraphInfo::kReplay;
      if (req->event) {
        req->event = create_event(q, index);
      }
    };

  } else if (role_ == Node::SERVER) {
    cmd_q()->reset_cmd = [](CommandQ* q, size_t index) {
      auto rsp = reinterpret_cast<AFTensorResponse*>(q->cmds[index]);
      rsp->capture_info.stage = CudaGraphInfo::kReplay;
      rsp->kv_meta.capture_info.stage = CudaGraphInfo::kReplay;
      if (rsp->event) {
        rsp->event = create_event(q, index);
      }
    };

  } else {
    cmd_q()->reset_cmd = [](void* cmd, size_t index) {};
  }
}
size_t CudaGraphContext::capture_end() {
  PS_LOG(INFO) << "capture end";
  // wait_capturing_complete();
  is_capturing_ = false;
  return reinterpret_cast<size_t>(cur_graph_.cmd_q);
}

void CudaGraphContext::wait_capturing_complete() {}

void CudaGraphContext::regist_flag(const std::vector<void*>& flags) {
  device_flag_buffer_ = flags;
  for (auto dptr : device_flag_buffer_) {
    host_flag_buffer_.push_back(Backend::Get()->GetAccessibleAddr(dptr, 0));
  }
}

}  // namespace ps
