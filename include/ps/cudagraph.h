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
  CommandQ(CudaGraphContext* ctx) : ctx_(ctx) {}

  std::vector<void*> cmds;
  std::vector<std::function<void(void*)>> invokers;
  std::function<void(void*)> reset_cmd;

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
  enum { kTerminate = -1 };

  CudaGraphContext() {}

  ~CudaGraphContext();

  static CudaGraphContext& instance();

  void set_config(int64_t num_micro_batch, Node::Role role) {
    num_micro_batch_ = num_micro_batch;
    role_ = role;
    if (role_ == Node::WORKER) {
      expect_flag_value_.resize(num_micro_batch, 1);
    } else if (role_ == Node::SERVER) {
      expect_flag_value_.resize(num_micro_batch, 2);
    }
  }

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
    CHECK(stage < host_flag_buffer_.size());

    // 使用per-stage的sync计数器作为layer_id，而不是全局的layer_id_
    // 这样可以确保每次sync使用正确的layer，不受主循环异步更新的影响
    int64_t sync_layer = sync_count_[stage];
    sync_count_[stage]++;

    int expected = expect_flag_value_[stage] + 2 * sync_layer;

    auto fetch_flag_values = [&]() {
      std::ostringstream oss;
      for (int64_t i = 0; i < num_micro_batch_; ++i) {
        oss << read_flag_from_device(i) << " ";
      }
      return oss.str();
    };

    // 使用cudaMemcpy从device读取flag，绕过host映射的缓存一致性问题
    int cur_value = read_flag_from_device(stage);
    int wait_count = 0;
    while (cur_value != expected) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(100));  // 改大到100ms验证PCIe读干扰假设
      cur_value = read_flag_from_device(stage);
      // 每1000次打印一次日志，避免日志过多
      if (wait_count++ % 1 == 0) {
        std::ostringstream oss;
        for (int64_t i = 0; i < num_micro_batch_; ++i) {
          oss << read_flag_from_device(i) << " ";
        }
        PS_LOG(INFO) << "stage:" << stage << " wait_flag:" << cur_value
                     << " expected:" << expected
                     << " flag_buffer:" << oss.str();
      }
    }
#if 0
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

  void regist_flag(const std::vector<void*>& flags) {
    device_flag_buffer_ = flags;
    for (auto it : device_flag_buffer_) {
      host_flag_buffer_.push_back(
          Backend::Get()->GetAccessibleAddr(device_flag_buffer_[stage], 0));
    }

    // 通过cudaMemcpy从device读取flag值
    // flag_buffer_ 现在直接存储 device 地址
    int read_flag_from_device(int stage) {
#if 0
    int value = 0;
    // CUDAGraph replay 在 captured stream 上执行，需要同步所有 stream
    // 确保 kernel 写入完成后再读取
    // flag_buffer_[stage] 现在直接是 device 地址，和 kernel 写入的地址一致
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
