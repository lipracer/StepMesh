#pragma once

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ps/base.h"
#include "ps/internal/message.h"
#include "ps/internal/threadsafe_queue.h"

#define USE_LOCKFREE
#define CG_DEBUG
#undef CG_DEBUG

// capture begin
// attn
// attn compute
// send message record && record rdma write flag
// launch wait kernel wait for flag

// ffn
// launch wait kernel wait for flag
// ffn compute
// send message record && record rdma write flag

// replay
// set batch size
// launch replay thread

namespace ps {

struct Event {
  enum Kind {
    kMemEvent,
    kCudaEvent,
    kGraphEventCapturing,
    kGraphEventReplaying
  };
  Kind kind;
  void* data;
};

#define N_EVENT(e) (reinterpret_cast<Event*>((e))->data)
#define W_EVENT(e) (reinterpret_cast<Event*>((e)))

class AFTensorRequest;
class AFTensorResponse;

enum class PipelineStage {
  kAttnSend,
  kAttnRecv,
  kFfnSend,
  kFfnRecv,
};

struct CudaGraphContext;

struct CommandQ {
  inline static constexpr size_t kPoolSize = 1024;

  CommandQ(CudaGraphContext* ctx) : ctx_(ctx) {
    tensor_events_ =
        reinterpret_cast<TensorEvent*>(malloc(sizeof(TensorEvent) * kPoolSize));
    native_events_ =
        reinterpret_cast<Event*>(malloc(sizeof(TensorEvent) * kPoolSize));
  }

  ~CommandQ() {
    free(tensor_events_);
    free(native_events_);
  }

  CommandQ(const CommandQ&) = delete;
  CommandQ& operator=(const CommandQ&) = delete;

  std::tuple<void*, void*> alloc_event();
  void free_event();

  std::vector<void*> cmds;
  std::vector<std::function<void(void*)>> invokers;
  TensorEvent* tensor_events_;
  Event* native_events_;
  std::atomic<size_t> pool_start_{0};
  std::atomic<size_t> pool_end_{0};

  std::function<void(CommandQ*, size_t)> reset_cmd;

  void record(void* cmd, std::function<void(void*)> invoker) {
    cmds.push_back(cmd);
    invokers.push_back(invoker);
  }

  void replay();
  CudaGraphContext* ctx_ = nullptr;
};

struct SMGraph {
  void* cmd_q;
};

struct CudaGraphContext {

  CudaGraphContext() {}

  ~CudaGraphContext();

  static CudaGraphContext& instance();

  void set_config(int64_t num_micro_batch, Node::Role role);

  void capture_begin();
  void destroy();

  CommandQ* cmd_q() { return reinterpret_cast<CommandQ*>(cur_graph_.cmd_q); }

  size_t capture_end();

  void wait_capturing_complete();

  void cudagraph_replay(size_t g) { replay(g); }

  template <typename CmdT, typename InvokerT>
  void record(CmdT& req, const InvokerT& invoker) {
    if (!is_capturing_) {
      return;
    }
    if constexpr (std::is_same<CmdT, AFTensorRequest>::value) {
      req.capture_info = ps::CudaGraphContext::instance().capture_info();
    } else if constexpr (std::is_same<CmdT, AFTensorResponse>::value) {
      req.capture_info = ps::CudaGraphContext::instance().capture_info();
      req.kv_meta.capture_info =
          ps::CudaGraphContext::instance().capture_info();
    }
    reinterpret_cast<CommandQ*>(cur_graph_.cmd_q)
        ->record(new CmdT(req), [invoker = invoker](void* rep) {
          invoker(*reinterpret_cast<CmdT*>(rep));
        });
  }

  bool is_capturing() const { return is_capturing_; }

  CudaGraphInfo capture_info() {
    PS_LOG(INFO) << "is_capturing_:" << is_capturing_;
    return CudaGraphInfo{
        is_capturing_ ? CudaGraphInfo::kCapturing : CudaGraphInfo::kDefault, 0};
  }

  void replay(size_t graph) {
    layer_id_ = 0;
    // 重置每个stage的sync计数器
    sync_count_.assign(num_micro_batch_, 0);
    cur_graph_.cmd_q = reinterpret_cast<void*>(graph);
    cmd_q()->replay();
  }

  // python side use it
  void* create_wait_event() {
    PS_LOG(INFO) << "create wait event stage:" << cur_stage_;
    return reinterpret_cast<int*>(host_flag_buffer_[cur_stage_]);
  }

  // python side use it
  int record(void* event, void* strem) { return 0; }

  int sync(void* event) {
    volatile int* flag = reinterpret_cast<volatile int*>(event);
    int stage = -1;
    for (auto f : host_flag_buffer_) {
      ++stage;
      if (f == event) {
        break;
      }
    }
    CHECK(static_cast<size_t>(stage) < host_flag_buffer_.size());

    int64_t sync_layer = sync_count_[stage];
    sync_count_[stage]++;

    int expected = expect_flag_value_[stage] + 2 * sync_layer;
#ifdef CG_DEBUG
    auto fetch_flag_values = [&]() {
      std::ostringstream oss;
      for (int64_t i = 0; i < num_micro_batch_; ++i) {
        oss << read_flag_from_device(i) << " ";
      }
      return oss.str();
    };
    int wait_count = 0;
#endif
    int cur_value = read_flag_from_device(stage);
    while (cur_value != expected) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(1)); 
      cur_value = read_flag_from_device(stage);
#ifdef CG_DEBUG
      if (wait_count++ % 1 == 0) {
        std::ostringstream oss;
        for (int64_t i = 0; i < num_micro_batch_; ++i) {
          oss << read_flag_from_device(i) << " ";
        }
        PS_LOG(INFO) << "stage:" << stage << " wait_flag:" << cur_value
                     << " expected:" << expected
                     << " flag_buffer:" << oss.str();
      }
#endif
    }
#ifdef CG_DEBUG
    std::ostringstream oss;
    oss << " data:";
    for (size_t i = 0; i < 16; ++i) {
      oss << reinterpret_cast<float*>(caches_[stage])[i] << " ";
    }
    PS_LOG(WARNING) << "cache:" << caches_[stage] << oss.str();
#endif
    return 0;
  }

  // capturing side use it: sync flag
  int event_sync(void* event) {
    // skip capturing stage, write kernel maybe not work
    PS_LOG(INFO) << "before event sync kind:" << W_EVENT(event)->kind;
    if (W_EVENT(event)->kind == Event::kGraphEventCapturing) {
      W_EVENT(event)->kind = Event::kGraphEventReplaying;
      PS_LOG(INFO) << "after event sync kind:" << W_EVENT(event)->kind;
      return 0;
    }
    return sync(N_EVENT(event));
  }

  int destroy_wait_event(void* e) {
    cmd_q()->free_event();
    return 0;
  }

  void regist_flag(const std::vector<void*>& flags);

  int read_flag_from_device(int stage) {
#if 0
    int value = 0;
    cudaMemcpy(&value, device_flag_buffer_[stage], sizeof(int),
               cudaMemcpyDeviceToHost);
    return value;

#else
    auto ptr = (volatile int*)(host_flag_buffer_[stage]);
    return *ptr;
#endif
  }

  void record_cache(void* cache) { caches_.push_back(cache); }

  void set_stage(int stage) {
    cur_stage_ = stage;
    PS_LOG(INFO) << "set stage:" << stage << " cur_stage_:" << cur_stage_;
  }

  std::atomic<bool> is_capturing_ = false;

  Node::Role role_ = Node::Role::JOINT;
  int64_t num_micro_batch_ = 0;

  std::atomic<int> cur_stage_ = -1;
  std::vector<void*> device_flag_buffer_;
  std::vector<void*> host_flag_buffer_;
  std::vector<int64_t> sync_count_;

  SMGraph cur_graph_;
  // for debug
  std::vector<void*> caches_;
  std::vector<int64_t> expect_flag_value_;
  int64_t layer_id_ = 0;
  };

}  // namespace ps
