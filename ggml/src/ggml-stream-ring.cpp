#include "ggml-stream-ring.hpp"
#include "ggml-stream-ring-bridge.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

#define CUDA_CHECK_RING(call)                                                       \
    do {                                                                            \
        cudaError_t err = (call);                                                   \
        if (err != cudaSuccess) {                                                   \
            std::fprintf(stderr, "[StreamRing] CUDA error %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err));              \
            return false;                                                           \
        }                                                                           \
    } while (0)

GgmlStreamRing & ggml_get_stream_ring() {
    static GgmlStreamRing instance;
    return instance;
}

GgmlStreamRing::~GgmlStreamRing() {
    shutdown();
}

bool GgmlStreamRing::init(const std::string & gguf_path,
                          const std::vector<StreamLayerPlan> & layers,
                          const std::vector<int> & device_ids,
                          StreamRamPreset preset) {
    shutdown();
    if (layers.empty() || device_ids.empty()) {
        return true;
    }

    fd_ = ::open(gguf_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        std::perror("[StreamRing] Failed to open GGUF file");
        return false;
    }

    layers_ = layers;
    int max_blk = 0;
    max_layer_bytes_ = 0;
    for (const auto & lp : layers_) {
        max_blk = std::max(max_blk, lp.layer_idx);
        max_layer_bytes_ = std::max(max_layer_bytes_, lp.total_host_bytes);
    }

    layer_to_plan_idx_.assign(max_blk + 1, -1);
    for (size_t i = 0; i < layers_.size(); ++i) {
        layer_to_plan_idx_[layers_[i].layer_idx] = static_cast<int>(i);
    }

    // Align slab stride to 4096 bytes for optimal NVMe/DMA alignment
    max_layer_bytes_ = (max_layer_bytes_ + 4095ULL) & ~4095ULL;

    size_t target_bytes = static_cast<size_t>(preset);
    size_t num_host_slots = std::max<size_t>(3, target_bytes / max_layer_bytes_);
    // Cap at total streamed layers so we don't allocate unused RAM if model has fewer layers
    num_host_slots = std::min(num_host_slots, layers_.size());
    host_arena_bytes_ = num_host_slots * max_layer_bytes_;

    std::fprintf(stderr, "[StreamRing] Allocating %.2f GB pinned host ring (%zu slots x %.2f MB)\n",
                 host_arena_bytes_ / (1024.0 * 1024.0 * 1024.0),
                 num_host_slots,
                 max_layer_bytes_ / (1024.0 * 1024.0));

    CUDA_CHECK_RING(cudaHostAlloc(&host_arena_, host_arena_bytes_, cudaHostAllocPortable));

    host_slots_.resize(num_host_slots);
    for (size_t s = 0; s < num_host_slots; ++s) {
        host_slots_[s].ptr = host_arena_ + (s * max_layer_bytes_);
    }

    // Initialize 3-slot VRAM staging ring on each active GPU
    gpu_rings_.resize(device_ids.size());
    for (size_t g = 0; g < device_ids.size(); ++g) {
        int dev = device_ids[g];
        gpu_rings_[g].device_id = dev;
        CUDA_CHECK_RING(cudaSetDevice(dev));
        CUDA_CHECK_RING(cudaStreamCreateWithFlags(&gpu_rings_[g].copy_stream, cudaStreamNonBlocking));

        size_t max_dev_bytes = 0;
        for (const auto & lp : layers_) {
            if (g < lp.gpu_slab_size.size()) {
                max_dev_bytes = std::max(max_dev_bytes, lp.gpu_slab_size[g]);
            }
        }
        max_dev_bytes = (max_dev_bytes + 4095ULL) & ~4095ULL;

        for (int k = 0; k < GPU_RING_SLOTS; ++k) {
            if (max_dev_bytes > 0) {
                CUDA_CHECK_RING(cudaMalloc(&gpu_rings_[g].vram_slots[k], max_dev_bytes));
            }
            CUDA_CHECK_RING(cudaEventCreateWithFlags(&gpu_rings_[g].dma_done[k], cudaEventDisableTiming));
            CUDA_CHECK_RING(cudaEventCreateWithFlags(&gpu_rings_[g].comp_done[k], cudaEventDisableTiming));
        }
    }

    running_.store(true);
    next_io_cursor_ = 0;
    io_thread_ = std::thread(&GgmlStreamRing::io_worker_loop, this);
    return true;
}

void GgmlStreamRing::shutdown() {
    if (running_.exchange(false)) {
        cv_io_.notify_all();
        cv_host_ready_.notify_all();
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
    }

    for (auto & gr : gpu_rings_) {
        cudaSetDevice(gr.device_id);
        for (int k = 0; k < GPU_RING_SLOTS; ++k) {
            if (gr.vram_slots[k]) cudaFree(gr.vram_slots[k]);
            if (gr.dma_done[k])   cudaEventDestroy(gr.dma_done[k]);
            if (gr.comp_done[k])  cudaEventDestroy(gr.comp_done[k]);
        }
        if (gr.copy_stream) cudaStreamDestroy(gr.copy_stream);
    }
    gpu_rings_.clear();

    if (host_arena_) {
        cudaFreeHost(host_arena_);
        host_arena_ = nullptr;
    }
    host_slots_.clear();

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool GgmlStreamRing::is_layer_streamed(int layer_idx) const {
    return layer_idx >= 0 &&
           static_cast<size_t>(layer_idx) < layer_to_plan_idx_.size() &&
           layer_to_plan_idx_[layer_idx] >= 0;
}

void GgmlStreamRing::io_worker_loop() {
    const size_t num_slots = host_slots_.size();
    const size_t total_plans = layers_.size();

    while (running_.load()) {
        int plan_idx = -1;
        size_t slot_idx = 0;

        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_io_.wait(lock, [&]() {
                if (!running_.load()) return true;
                size_t candidate_slot = static_cast<size_t>(next_io_cursor_) % num_slots;
                // Slot is free to overwrite if not currently holding an unconsumed layer or active DMA
                return host_slots_[candidate_slot].dma_refs == 0 &&
                       !host_slots_[candidate_slot].ready;
            });

            if (!running_.load()) break;

            plan_idx = next_io_cursor_;
            slot_idx = static_cast<size_t>(plan_idx) % num_slots;
            next_io_cursor_ = (next_io_cursor_ + 1) % static_cast<int>(total_plans);
        }

        const auto & plan = layers_[plan_idx];
        uint8_t * dst_base = host_slots_[slot_idx].ptr;

        // Coalesced sequential pread of all overflow slices for this layer
        for (const auto & slice : plan.slices) {
            size_t bytes_read = 0;
            while (bytes_read < slice.byte_size && running_.load()) {
                ssize_t n = ::pread(fd_,
                                    dst_base + slice.slab_offset + bytes_read,
                                    slice.byte_size - bytes_read,
                                    static_cast<off_t>(slice.file_offset + bytes_read));
                if (n <= 0) break;
                bytes_read += static_cast<size_t>(n);
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);
            host_slots_[slot_idx].loaded_layer = plan.layer_idx;
            host_slots_[slot_idx].ready = true;
        }
        cv_host_ready_.notify_all();
    }
}

void GgmlStreamRing::enqueue_dma_if_ready(int plan_idx) {
    const auto & plan = layers_[plan_idx];
    const size_t h_slot = static_cast<size_t>(plan_idx) % host_slots_.size();
    const int    g_slot = plan_idx % GPU_RING_SLOTS;

    std::unique_lock<std::mutex> lock(mtx_);
    cv_host_ready_.wait(lock, [&]() {
        return !running_.load() ||
               (host_slots_[h_slot].ready && host_slots_[h_slot].loaded_layer == plan.layer_idx);
    });
    if (!running_.load()) return;

    for (size_t g = 0; g < gpu_rings_.size(); ++g) {
        auto & gr = gpu_rings_[g];
        if (gr.slot_layer[g_slot] == plan.layer_idx) continue; // Already DMA'd

        cudaSetDevice(gr.device_id);
        // Ensure prior compute using this VRAM slot has finished before overwriting
        cudaStreamWaitEvent(gr.copy_stream, gr.comp_done[g_slot], 0);

        size_t copy_bytes = plan.gpu_slab_size[g];
        if (copy_bytes > 0) {
            cudaMemcpyAsync(gr.vram_slots[g_slot],
                            host_slots_[h_slot].ptr + plan.gpu_slab_offset[g],
                            copy_bytes,
                            cudaMemcpyHostToDevice,
                            gr.copy_stream);
        }
        cudaEventRecord(gr.dma_done[g_slot], gr.copy_stream);
        gr.slot_layer[g_slot] = plan.layer_idx;
    }
}

void GgmlStreamRing::begin_layer(int layer_idx, cudaStream_t compute_stream, int device_id) {
    int plan_idx = layer_to_plan_idx_[layer_idx];
    if (plan_idx < 0) return;

    // 1. Ensure current layer DMA is scheduled and make compute_stream wait on dma_done
    enqueue_dma_if_ready(plan_idx);

    const int g_slot = plan_idx % GPU_RING_SLOTS;
    for (auto & gr : gpu_rings_) {
        if (gr.device_id == device_id) {
            cudaStreamWaitEvent(compute_stream, gr.dma_done[g_slot], 0);
            break;
        }
    }

    // 2. Lookahead: kick off DMA for (plan_idx + 1) if its host slot is already loaded
    int next_plan = (plan_idx + 1) % static_cast<int>(layers_.size());
    size_t next_h_slot = static_cast<size_t>(next_plan) % host_slots_.size();
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (host_slots_[next_h_slot].ready &&
            host_slots_[next_h_slot].loaded_layer == layers_[next_plan].layer_idx) {
            lock.~lock_guard();
            enqueue_dma_if_ready(next_plan);
        }
    }
}

void GgmlStreamRing::end_layer(int layer_idx, cudaStream_t compute_stream, int device_id) {
    int plan_idx = layer_to_plan_idx_[layer_idx];
    if (plan_idx < 0) return;

    const int g_slot = plan_idx % GPU_RING_SLOTS;
    for (auto & gr : gpu_rings_) {
        if (gr.device_id == device_id) {
            cudaEventRecord(gr.comp_done[g_slot], compute_stream);
            break;
        }
    }

    // Release host slot once the last GPU in the split finishes queuing-up this layer
    if (device_id == gpu_rings_.back().device_id) {
        size_t h_slot = static_cast<size_t>(plan_idx) % host_slots_.size();

        {
            std::lock_guardstd::mutex lock(mtx_);
            host_slots_[h_slot].ready = false;
        }
        cv_io_.notify_one();
    }
}

void * GgmlStreamRing::get_device_ptr(int layer_idx, int device_id, size_t slab_offset) const {
    int plan_idx = layer_to_plan_idx_[layer_idx];
    if (plan_idx < 0) return nullptr;

    const int g_slot = plan_idx % GPU_RING_SLOTS;
    for (size_t g = 0; g < gpu_rings_.size(); ++g) {
        if (gpu_rings_[g].device_id == device_id) {
            size_t rel_offset = slab_offset - layers_[plan_idx].gpu_slab_offset[g];
            return gpu_rings_[g].vram_slots[g_slot] + rel_offset;
        }
    }
    
    return nullptr;
}

static bool stream_ring_init_trampoline(const std::string & gguf_path,
                                        const std::vector<StreamLayerPlan> & layers,
                                        const std::vector<int> & device_ids,
                                        StreamRamPreset preset) {
    return ggml_get_stream_ring().init(gguf_path, layers, device_ids, preset);
}

namespace {
struct StreamRingAutoRegister {
    StreamRingAutoRegister() {
        ggml_register_stream_ring_init(&stream_ring_init_trampoline);
    }
};
static StreamRingAutoRegister g_stream_ring_auto_register;
} // namespace
