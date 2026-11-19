# Clean previous build cache (if any)
rm -rf build

# Reconfigure NVCC environment variables + CMake flags
# IMPORTANT - ensure you update CUDA directory to match your environment!
PATH=$PATH:/usr/local/cuda-12.8/bin

# Force the NVCC to shut-up about compute_70 deprecation
# (Comment out if you're using modern NVIDIA Architecture)
# ... Then send one of those bad-boys my way! 
# Lol sorry <3 <3 <3 (Kidding)
export NVCC_PREPEND_FLAGS="-Wno-deprecated-gpu-targets"

# Cont... (if building for other archs use: 50;61;86;89)
# This is in regards to "DCMAKE_CUDA_ARCHITECTURES" definitions
# IMPORTANT - ensure you update CUDA directory to match your environment!
NVCC_CCBIN=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13 CC=gcc-13 CXX=g++-13 \
cmake -B build \
  -DCMAKE_UNITY_BUILD=OFF \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NCCL=OFF \
  -DGGML_OPENCL=OFF \
  -DGGML_VULKAN=OFF \
  -DCMAKE_C_COMPILER=gcc-13 \
  -DCMAKE_CXX_COMPILER=g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="70;70-real;70-virtual" \
  -DCMAKE_BUILD_TYPE=Release

#Compile targeted llama-server binary
cmake --build build --config Release --target llama-server ggml ggml-base ggml-cuda llama-gguf-split llama-quantize llama-tts mtmd -j 11