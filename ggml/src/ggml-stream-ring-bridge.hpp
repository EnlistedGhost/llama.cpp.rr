#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class StreamRamPreset : size_t {
    RAM_16GB = 16ULL * 1024 * 1024 * 1024,
    RAM_32GB = 32ULL * 1024 * 1024 * 1024,
    RAM_64GB = 64ULL * 1024 * 1024 * 1024,
};

struct StreamTensorSlice {
    uint64_t file_offset;
    size_t   byte_size;
    size_t   slab_offset;
    int      device_id;
};

struct StreamLayerPlan {
    int    layer_idx;
    size_t total_host_bytes;
    std::vector<StreamTensorSlice> slices;
    std::vector<size_t> gpu_slab_offset;
    std::vector<size_t> gpu_slab_size;
};

using StreamRingInitFn = bool (*)(const std::string &,
                                  const std::vector<StreamLayerPlan> &,
                                  const std::vector<int> &,
                                  StreamRamPreset);

void ggml_register_stream_ring_init(StreamRingInitFn fn);
bool ggml_invoke_stream_ring_init(const std::string & gguf_path,
                                  const std::vector<StreamLayerPlan> & layers,
                                  const std::vector<int> & device_ids,
                                  StreamRamPreset preset);
