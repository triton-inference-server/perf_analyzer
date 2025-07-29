<!--
Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Recommended Installation Method

## Triton SDK Container

The recommended way to access Perf Analyzer is to run the pre-built executable
from within the Triton SDK docker container available on the
[NVIDIA GPU Cloud Catalog](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver).
As long as the SDK container has its network exposed to the address and port of
the inference server, Perf Analyzer will be able to run.

```bash
export RELEASE=<yy.mm> # e.g. to use the release from the end of December of 2024, do `export RELEASE=24.12`

docker run --rm --gpus=all -it --net=host nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# inside container
perf_analyzer -m <model>
```

# Alternative Installation Methods

- [pip](#pip)
- [Build from Source](#build-from-source)

## pip

```bash
pip install perf-analyzer

perf_analyzer -m <model>
```

**Warning**: If any runtime dependencies are missing, Perf Analyzer will produce
errors showing which ones are missing. You will need to manually install them.

## Build from Source

```bash
docker run --rm --gpus all -it --network host ubuntu:24.04

# inside container, install build/runtime dependencies
apt update && apt install -y curl

curl -LsSf https://apt.kitware.com/kitware-archive.sh | sh

CMAKE_VERSION_FULL=$(apt-cache madison cmake | awk '/3.31.8/ {print $3; exit}')

apt update && DEBIAN_FRONTEND=noninteractive apt install -y cmake=${CMAKE_VERSION_FULL} cmake-data=${CMAKE_VERSION_FULL} g++ git libssl-dev nvidia-cuda-toolkit python3 rapidjson-dev zlib1g-dev

git clone --depth 1 https://github.com/triton-inference-server/perf_analyzer.git

mkdir perf_analyzer/build

cmake -B perf_analyzer/build -S perf_analyzer

cmake --build perf_analyzer/build --parallel 8

export PATH=$(pwd)/perf_analyzer/build/perf_analyzer/src/perf-analyzer-build${PATH:+:${PATH}}

perf_analyzer -m <model>
```

- To enable
  [OpenAI mode](benchmarking.md#benchmarking-openai), add
  `-D TRITON_ENABLE_PERF_ANALYZER_OPENAI=ON` to the first `cmake` command.
- To enable
  [C API mode](benchmarking.md#benchmarking-triton-directly-via-c-api), add
  `-D TRITON_ENABLE_PERF_ANALYZER_C_API=ON` to the first `cmake` command.
- To enable [TorchServe backend](benchmarking.md#benchmarking-torchserve), add
  `-D TRITON_ENABLE_PERF_ANALYZER_TS=ON` to the first `cmake` command.
- To enable
  [Tensorflow Serving backend](benchmarking.md#benchmarking-tensorflow-serving),
  add `-D TRITON_ENABLE_PERF_ANALYZER_TFS=ON` to the first `cmake` command.
- To disable
  [CUDA shared memory support](input_data.md#shared-memory) and the dependency
  on CUDA toolkit libraries, add
  `-D TRITON_ENABLE_GPU=OFF` to the first `cmake` command.
