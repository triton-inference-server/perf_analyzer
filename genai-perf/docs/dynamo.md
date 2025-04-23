<!--
Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Benchmark [Dynamo](https://github.com/ai-dynamo/dynamo) with GenAI-Perf

NVIDIA Dynamo is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments. This tutorial demonstrates how to use GenAI-Perf to benchmark the performance of Dynamo.

### Table of Contents

- [Build Dynamo](#build)
- [Benchmark Dynamo with GenAI-Perf](#benchmark)

</br>

## Build Dynamo <a id="build"></a>

Build Dynamo and install sglang using the following commands
```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libssl-dev libclang-dev protobuf-compiler python3-dev cmake

./container/build.sh
./container/run.sh -it --mount-workspace

cargo build --release
mkdir -p /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/http /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/llmctl /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/dynamo-run /workspace/deploy/dynamo/sdk/src/dynamo/sdk/cli/bin

uv pip install -e .
```

```bash
uv venv
source .venv/bin/activate
uv pip install pip
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```


## Benchmark Dynamo with GenAI-Perf <a id="benchmark"></a>

### Option1: Start Server using dynamo-run

```bash
dynamo-run in=http out=sglang deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

### Run GenAI-Perf

Run GenAI-Perf in another terminal:

```bash
genai-perf profile \
  -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 128 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --url localhost:8080 \
  --streaming \
  --request-count 10 \
  --warmup-request-count 2
```

Example output:

```
                                     NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                            Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│             Time To First Token (ms) │    39.13 │    21.23 │    41.49 │    41.48 │    41.32 │    41.29 │
│            Time To Second Token (ms) │    14.50 │    13.51 │    17.50 │    17.26 │    15.14 │    14.45 │
│                 Request Latency (ms) │ 1,799.53 │ 1,783.41 │ 1,802.41 │ 1,802.39 │ 1,802.18 │ 1,801.73 │
│             Inter Token Latency (ms) │    17.96 │    17.95 │    17.98 │    17.98 │    17.97 │    17.97 │
│     Output Token Throughput Per User │    55.67 │    55.61 │    55.71 │    55.70 │    55.70 │    55.69 │
│                    (tokens/sec/user) │          │          │          │          │          │          │
│      Output Sequence Length (tokens) │    99.00 │    99.00 │    99.00 │    99.00 │    99.00 │    99.00 │
│       Input Sequence Length (tokens) │   128.00 │   128.00 │   128.00 │   128.00 │   128.00 │   128.00 │
│ Output Token Throughput (tokens/sec) │    54.52 │      N/A │      N/A │      N/A │      N/A │      N/A │
│         Request Throughput (per sec) │     0.55 │      N/A │      N/A │      N/A │      N/A │      N/A │
│                Request Count (count) │    10.00 │      N/A │      N/A │      N/A │      N/A │      N/A │
└──────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### Option2: Start Server using dynamo serve

Ensure you have NATS and etcd running before starting the server.

```bash
cd deploy
docker compose up -d
```

```bash
cd examples/llm
dynamo serve graphs.agg:Frontend -f ./configs/agg.yaml 
```

### Run GenAI-Perf

Run GenAI-Perf in another terminal:

```bash
genai-perf profile \
  -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 128 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 20 \
  --output-tokens-stddev 0 \
  --streaming \
  --request-count 10 \
  --warmup-request-count 2
```

Example output:

```
                               NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                            Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│             Time To First Token (ms) │ 140.18 │ 120.43 │ 264.11 │ 253.22 │ 155.25 │ 130.56 │
│            Time To Second Token (ms) │  19.32 │  17.84 │  20.58 │  20.58 │  20.54 │  20.43 │
│                 Request Latency (ms) │ 530.43 │ 510.31 │ 654.62 │ 643.89 │ 547.32 │ 521.66 │
│             Inter Token Latency (ms) │  20.54 │  20.27 │  20.82 │  20.81 │  20.66 │  20.62 │
│     Output Token Throughput Per User │  48.69 │  48.03 │  49.34 │  49.32 │  49.08 │  48.92 │
│                    (tokens/sec/user) │        │        │        │        │        │        │
│      Output Sequence Length (tokens) │  20.00 │  20.00 │  20.00 │  20.00 │  20.00 │  20.00 │
│       Input Sequence Length (tokens) │ 128.00 │ 128.00 │ 128.00 │ 128.00 │ 128.00 │ 128.00 │
│ Output Token Throughput (tokens/sec) │  37.69 │    N/A │    N/A │    N/A │    N/A │    N/A │
│         Request Throughput (per sec) │   1.88 │    N/A │    N/A │    N/A │    N/A │    N/A │
│                Request Count (count) │  10.00 │    N/A │    N/A │    N/A │    N/A │    N/A │
└──────────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```
