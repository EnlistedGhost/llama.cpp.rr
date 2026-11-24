#include "ggml-stream-ring-bridge.hpp"

static StreamRingInitFn g_stream_ring_init_fn = nullptr;

void ggml_register_stream_ring_init(StreamRingInitFn fn) {
    g_stream_ring_init_fn = fn;
}

bool ggml_invoke_stream_ring_init(const std::string & gguf_path,
                                  const std::vector<StreamLayerPlan> & layers,
                                  const std::vector<int> & device_ids,
                                  StreamRamPreset preset) {
    if (!g_stream_ring_init_fn) {
        return false;
    }
    return g_stream_ring_init_fn(gguf_path, layers, device_ids, preset);
}
