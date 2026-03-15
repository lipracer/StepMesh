/**
 *  Copyright (C) by StepAI Contributors. 2025.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <sys/mman.h>

#include <mutex>

#include "dmlc/backend_registry.h"
#include "ps/backend.h"
#include "ps/cudagraph.h"
#include "ps/hash_table8.hpp"
#include "ps/internal/gpu_backend.h"

#define KLX_RT_CALL(func, ...)                                       \
  do {                                                               \
    auto klx_errno = func(__VA_ARGS__);                              \
    PS_CHECK_EQ(klx_errno, 0)                                        \
        << #func << " failed err:" << cudaGetErrorString(klx_errno); \
  } while (0)

#define CHECK_MEM_LEAK

namespace klx {

using namespace ps;

#ifdef CHECK_MEM_LEAK
static struct MemChecker {
  std::unordered_map<void*, size_t> mem_info_;

  ~MemChecker() {
    PS_LOG(INFO) << "MemChecker destruct!";
    if (!mem_info_.empty()) {
      for (const auto& it : mem_info_) {
        PS_LOG(INFO) << "leak mem:" << it.first << " size:" << it.second;
      }
    }
  }
} mem_checker;
#endif

class KlxBackend : public Backend {
 public:
  KlxBackend();
  int SetDevice(int dev) override;
  int GetDeviceId() override;
  at::Device GetDevice() override;
  void* Alloc(uint64_t size) override;
  void Free(void* m) override;
  void* CreateEvent() override;
  int FreeEvent(void* event) override;
  int RecordEvent(void* event, void* stream) override;
  int SyncEvent(void* event) override;

  void* GetAccessibleAddr(void* devicePtr, size_t size) final;

  void* GetAccessibleAddr(const at::Tensor& tensor) final;

  void* GetDeviceAddrFromHostPtr(void* hostPtr, size_t size) final;

 private:
  void* CreateCudaEvent();
  int FreeCudaEvent(void* event);
  int RecordCudaEvent(void* event, void* stream);
  int SyncCudaEvent(void* event);

  void* CreateMemEvent();
  int FreeMemEvent(void* event);
  int RecordMemEvent(void* event, void* stream);
  int SyncMemEvent(void* event);

 private:
  inline void DoInitGpu() {
    static thread_local int gpu_idx = -1;
    if (gpu_idx == -1) {
      PS_CHECK_GE(gpu_idx_, 0)
          << "cannot set device " << gpu_idx_ << " for klx backend";
      SetDevice(gpu_idx_);
      gpu_idx = gpu_idx_;
    }
  }

  /** \brief for cpu backend, the device stands for numa id */
  int gpu_idx_ = -1;
  int mem_sync_ = 1;
  // host address to device address map
  std::mutex mtx_;
  emhash8::HashMap<void*, void*> ha_da_map_;
};

KlxBackend::KlxBackend() {
  Environment::Get()->find("STEPMESH_MEM_SYNC", &mem_sync_, mem_sync_);
  PS_LOG(INFO) << "create klx backend";
}

int KlxBackend::SetDevice(int dev) {
  static thread_local int max_num_dev = GetEnv("MAX_NUM_DEVICES_PER_NODE", 7);
  PS_CHECK_GE(dev, 0) << "cannot set dev=" << dev << " for klx backend";
  PS_CHECK_LE(dev, max_num_dev)
      << "cannot set dev=" << dev << " for klx backend";
  static thread_local int gpu_idx = -1;

  gpu_idx_ = dev;
  if (gpu_idx == -1 || gpu_idx != gpu_idx_) {
    gpu_idx = gpu_idx_;
    KLX_RT_CALL(cudaSetDevice, gpu_idx_);
  }

  return BACKEND_OK;
}

int KlxBackend::GetDeviceId() {
  static thread_local int gpu_idx = -1;
  if (gpu_idx != -1) {
    KLX_RT_CALL(cudaGetDevice, &gpu_idx);
  }
  return gpu_idx;
}

at::Device KlxBackend::GetDevice() {
  PS_CHECK_GE(gpu_idx_, 0) << "device index is not initialized for klx backend";
  return {at::kCUDA, static_cast<char>(gpu_idx_)};
}

void* KlxBackend::Alloc(uint64_t size) {
  DoInitGpu();
  void* ptr = nullptr;
  KLX_RT_CALL(cudaMalloc, &ptr, size);
  auto hostPtr = GetAccessibleAddr(ptr, size);
#ifdef CHECK_MEM_LEAK
  mem_checker.mem_info_[ptr] = size;
#endif
  return hostPtr;
}

void KlxBackend::Free(void* m) {
  PS_CHECK_NE(m, nullptr) << "backend cannot free null memory";
  PS_VLOG(3) << "free klx memory " << m;
  {
    std::lock_guard<std::mutex> lg(mtx_);
#ifdef CHECK_MEM_LEAK
    mem_checker.mem_info_.erase(m);
#endif

    if (ha_da_map_.erase(m)) {
      m = ha_da_map_[m];
    }
  }
  KLX_RT_CALL(cudaFree, m);
}

void* KlxBackend::GetAccessibleAddr(void* devicePtr, size_t size) {
  struct cudaPointerAttributes attrs;
  KLX_RT_CALL(cudaPointerGetAttributes, &attrs, devicePtr);
  PS_LOG(INFO) << "GetAccessibleAddr devicePtr=" << devicePtr
               << " hostPtr=" << attrs.hostPointer;
  std::lock_guard<std::mutex> lg(mtx_);
  if (ha_da_map_.find(attrs.hostPointer) != ha_da_map_.end()) {
    return reinterpret_cast<char*>(attrs.hostPointer) +
           (reinterpret_cast<intptr_t>(devicePtr) -
            reinterpret_cast<intptr_t>(ha_da_map_[attrs.hostPointer]));
  }
  ha_da_map_.emplace_unique(attrs.hostPointer, devicePtr);
  PS_CHECK_NE(attrs.hostPointer, nullptr);
  return attrs.hostPointer;
}

void* KlxBackend::GetAccessibleAddr(const at::Tensor& tensor) {
  if (tensor.device().type() == at::kCUDA) {
    return GetAccessibleAddr(
        reinterpret_cast<char*>(tensor.data_ptr()) + tensor.storage_offset(),
        tensor.numel() * tensor.element_size());
  }
  PS_CHECK_EQ(tensor.device().type(), at::kCPU);
  return reinterpret_cast<char*>(tensor.data_ptr()) + tensor.storage_offset();
}

void* KlxBackend::GetDeviceAddrFromHostPtr(void* hostPtr, size_t size) {
  std::lock_guard<std::mutex> lg(mtx_);
  PS_CHECK_NE(ha_da_map_.find(hostPtr), ha_da_map_.end());
  return ha_da_map_[hostPtr];
}

void* KlxBackend::CreateEvent() {
  DoInitGpu();
  Event* e = (Event*)malloc(sizeof(Event));
  if (ps::CudaGraphContext::instance().is_capturing()) {
    e->data = CudaGraphContext::instance().create_wait_event();
    e->kind = Event::kGraphEventCapturing;
  } else if (!mem_sync_) {
    e->data = CreateCudaEvent();
    e->kind = Event::kCudaEvent;
  } else {
    e->data = CreateMemEvent();
    e->kind = Event::kMemEvent;
  }
  return e;
}

int KlxBackend::FreeEvent(void* event) {
  DoInitGpu();
  PS_CHECK_NE(event, nullptr) << "backend cannot free null event";
  auto e = event;
  event = N_EVENT(event);
  free(e);
  if (!mem_sync_) {
    return FreeCudaEvent(N_EVENT(event));
  } else {
    return FreeMemEvent(N_EVENT(event));
  }
}

int KlxBackend::RecordEvent(void* event, void* stream) {
  DoInitGpu();
  PS_LOG(INFO) << "klx backend record event kind:" << W_EVENT(event)->kind
               << " klx backend record event:" << event;
  if (W_EVENT(event)->kind >= Event::kGraphEventCapturing) {
    return CudaGraphContext::instance().record(N_EVENT(event), stream);
  }

  PS_CHECK_NE(event, nullptr) << "backend cannot record null event";
  if (!mem_sync_) {
    return RecordCudaEvent(N_EVENT(event), stream);
  } else {
    return RecordMemEvent(N_EVENT(event), stream);
  }
}

int KlxBackend::SyncEvent(void* event) {
  DoInitGpu();
  PS_LOG(INFO) << "klx backend sync event kind:" << W_EVENT(event)->kind;
  PS_CHECK_NE(event, nullptr) << "backend cannot sync null event";
  if (W_EVENT(event)->kind >= Event::kGraphEventCapturing) {
    return CudaGraphContext::instance().event_sync(event);
  }

  if (!mem_sync_) {
    return SyncCudaEvent(N_EVENT(event));
  } else {
    return SyncMemEvent(N_EVENT(event));
  }
}

void* KlxBackend::CreateCudaEvent() {
  cudaEvent_t* ev = nullptr;
  cudaMallocHost(&ev, sizeof(cudaEvent_t));
  auto status = cudaEventCreateWithFlags(ev, cudaEventDisableTiming);
  PS_CHECK_EQ(status, cudaSuccess)
      << "cudaEventCreateWithFlags failed for klx " << gpu_idx_;
  return reinterpret_cast<void*>(ev);
}

int KlxBackend::FreeCudaEvent(void* event) {
  auto ev = reinterpret_cast<cudaEvent_t*>(event);
  cudaError_t err = cudaEventDestroy(*ev);
  PS_CHECK_EQ(err, cudaSuccess)
      << "cudaEventDestroy failed for event " << reinterpret_cast<void*>(event)
      << " (" << cudaGetErrorString(err) << ")";
  cudaFreeHost(ev);
  return BACKEND_OK;
}

int KlxBackend::RecordCudaEvent(void* event, void* stream) {
  cudaStream_t cuda_stream;
  if (stream == nullptr) {
    cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  } else {
    cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  }

  auto ev = reinterpret_cast<cudaEvent_t*>(event);
  auto status = cudaEventRecord(*ev, cuda_stream);
  if (status == cudaSuccess) {
    return BACKEND_OK;
  } else {
    PS_LOG(WARNING) << "failed to record klx event: "
                    << " (" << cudaGetErrorString(status) << ")";
    return BACKEND_FAILED;
  }
}

int KlxBackend::SyncCudaEvent(void* event) {
  auto ev = reinterpret_cast<cudaEvent_t*>(event);
  cudaError_t status;
  while (true) {
    status = cudaEventQuery(*ev);
    if (status == cudaErrorNotReady) {
      sched_yield();
      continue;
    }
    break;
  }
  if (status != cudaSuccess) {
    PS_LOG(WARNING) << "failed to sync klx event: "
                    << " (" << cudaGetErrorString(status) << ")";
    return BACKEND_FAILED;
  }

  return BACKEND_OK;
}

struct KlxBackendMemEvent {
  int* gpu_flag = nullptr;
  int* cpu_flag = nullptr;
};

void* KlxBackend::CreateMemEvent() {
  struct KlxBackendMemEvent* ev = nullptr;
  AT_CUDA_CHECK(cudaMallocHost(&ev, sizeof(KlxBackendMemEvent)));
  AT_CUDA_CHECK(cudaMalloc(&(ev->gpu_flag), sizeof(int)));
  AT_CUDA_CHECK(cudaMemset(ev->gpu_flag, 0, sizeof(int)));
  AT_CUDA_CHECK(
      cudaMallocHost(reinterpret_cast<void**>(&(ev->cpu_flag)), sizeof(int)));
  *ev->cpu_flag = 0;
  return reinterpret_cast<void*>(ev);
}

int KlxBackend::FreeMemEvent(void* event) {
  auto ev = reinterpret_cast<KlxBackendMemEvent*>(event);
  AT_CUDA_CHECK(cudaFree(ev->gpu_flag));
  AT_CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(ev->cpu_flag)));
  AT_CUDA_CHECK(cudaFreeHost(ev));
  return BACKEND_OK;
}

int KlxBackend::RecordMemEvent(void* event, void* stream) {
  auto ev = reinterpret_cast<KlxBackendMemEvent*>(event);
  *(ev->cpu_flag) = 1;
  cudaStream_t cuda_stream;
  if (stream == nullptr) {
    cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  } else {
    cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  }

  AT_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(ev->cpu_flag),
                                ev->gpu_flag, sizeof(int),
                                cudaMemcpyDeviceToHost, cuda_stream));
  return BACKEND_OK;
}

int KlxBackend::SyncMemEvent(void* event) {
  auto ev = reinterpret_cast<KlxBackendMemEvent*>(event);
  while (*(ev->cpu_flag) == 1) {
    _mm_pause();
  }
  return BACKEND_OK;
}

dmlc::backend_registry<KlxBackend> _("GPU");

}  // namespace klx
