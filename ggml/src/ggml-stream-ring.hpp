#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward-declare CUDA stream pointer so callers (like libllama) don't need CUDA headers
struct CUstream_st;
typedef struct CUstream_st * cudaStream_t;

class GgmlStreamRing {
public:
    GgmlStreamRing() = default;
    ~GgmlStreamRing();

    // Initialize pinned host arena (16/32/64 GB) and 3-slot VRAM staging rings across active GPUs
    bool init(const std::string & gguf_path,
              const std::vector<StreamLayerPlan> & layers,
              const std::vector<int> & device_ids,
              StreamRamPreset preset = StreamRamPreset::RAM_16GB);

    void shutdown();

    // Called by ggml-cuda right before executing layer_idx:
    // Ensures layer_idx DMA is complete on all GPUs and schedules lookahead DMA + SSD reads.
    void begin_layer(int layer_idx, cudaStream_t compute_stream, int device_id);

    // Called by ggml-cuda right after layer_idx kernels are enqueued on compute_stream:
    // Records compute_done event so the VRAM/host slot can be safely recycled.
    void end_layer(int layer_idx, cudaStream_t compute_stream, int device_id);

    // Returns the live VRAM pointer for a streamed tensor during layer execution
    void * get_device_ptr(int layer_idx, int device_id, size_t slab_offset) const;

    bool is_layer_streamed(int layer_idx) const;

private:
    static constexpr int GPU_RING_SLOTS = 3; // Triple-buffered VRAM staging per GPU

    struct HostSlot {
        uint8_t * ptr = nullptr;
        int       loaded_layer = -1;
        bool      ready = false;
        int       dma_refs = 0; // Active GPU copy streams reading from this host slot
    };

    struct GpuDeviceRing {
        int          device_id = 0;
        cudaStream_t copy_stream = nullptr;
        uint8_t *    vram_slots[GPU_RING_SLOTS] = {nullptr, nullptr, nullptr};
        void *       dma_done[GPU_RING_SLOTS]   = {nullptr, nullptr, nullptr};
        void *       comp_done[GPU_RING_SLOTS]  = {nullptr, nullptr, nullptr};
        int          slot_layer[GPU_RING_SLOTS] = {-1, -1, -1};
    };

    void io_worker_loop();
    void enqueue_dma_if_ready(int target_layer);

    int                          fd_ = -1;
    std::vector<StreamLayerPlan> layers_;
    std::vector<int>             layer_to_plan_idx_; // Maps model blk.i -> index in layers_ (or -1)

    uint8_t *                    host_arena_ = nullptr;
    size_t                       host_arena_bytes_ = 0;
    size_t                       max_layer_bytes_ = 0;
    std::vector<HostSlot>        host_slots_;
    std::vector<GpuDeviceRing>   gpu_rings_;

    std::thread                  io_thread_;
    std::mutex                   mtx_;
    std::condition_variable      cv_io_;
    std::condition_variable      cv_host_ready_;
    std::atomic<bool>            running_{false};
    int                          next_io_cursor_ = 0; // Index into layers_ to prefetch next
};

// Global singleton accessor for ggml-cuda / llama-model integration
GgmlStreamRing & ggml_get_stream_ring();
