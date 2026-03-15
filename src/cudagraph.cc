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
    reset_cmd(cmds[i]);
    reset_cmd(cmds[i + 1]);
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

CudaGraphContext::~CudaGraphContext() {}

void CudaGraphContext::destroy() {}

CudaGraphContext& CudaGraphContext::instance() {
  static CudaGraphContext instance;
  return instance;
}

void CudaGraphContext::capture_begin() {
  PS_LOG(INFO) << "capture begin";
  is_capturing_ = true;
  cur_graph_ = SMGraph{.cmd_q = new CommandQ(this)};

  auto create_event = [this]() {
    Event* e = (Event*)malloc(sizeof(Event));
    e->data = CudaGraphContext::instance().create_wait_event();
    e->kind = Event::kGraphEventReplaying;
    auto ne = new TensorEvent(e);
    ne->Record();
    return ne;
  };

  if (role_ == Node::WORKER) {
    cmd_q()->reset_cmd = [create_event](void* cmd) {
      auto req = reinterpret_cast<AFTensorRequest*>(cmd);
      req->capture_info.stage = CudaGraphInfo::kReplay;
      if (req->event) {
        req->event = create_event();
      }
    };

  } else if (role_ == Node::SERVER) {
    cmd_q()->reset_cmd = [create_event](void* cmd) {
      auto rsp = reinterpret_cast<AFTensorResponse*>(cmd);
      rsp->capture_info.stage = CudaGraphInfo::kReplay;
      rsp->kv_meta.capture_info.stage = CudaGraphInfo::kReplay;
      if (rsp->event) {
        rsp->event = create_event();
      }
    };

  } else {
    cmd_q()->reset_cmd = [](void* cmd) {};
  }
}

size_t CudaGraphContext::capture_end() {
  PS_LOG(INFO) << "capture end";
  // wait_capturing_complete();
  is_capturing_ = false;
  return reinterpret_cast<size_t>(cur_graph_.cmd_q);
}

void CudaGraphContext::wait_capturing_complete() {}

}  // namespace ps
