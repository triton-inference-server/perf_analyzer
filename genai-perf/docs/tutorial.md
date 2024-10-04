<!--
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Profile Large Language Models with GenAI-Perf

This tutorial will demonstrate how you can use GenAI-Perf to measure the performance of
various inference endpoints such as
[KServe inference protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
and [OpenAI API](https://platform.openai.com/docs/api-reference/introduction)
that are widely used across the industry.

### Table of Contents

- [Profile GPT2 running on Triton + TensorRT-LLM Backend](#tensorrt-llm)
- [Profile GPT2 running on Triton + vLLM Backend](#triton-vllm)
- [Profile Zephyr-7B-Beta running on OpenAI Chat Completions API-Compatible Server](#openai-chat)
- [Profile GPT2 running on OpenAI Completions API-Compatible Server](#openai-completions)

</br>

## Profile GPT-2 running on Triton + TensorRT-LLM <a id="tensorrt-llm"></a>

You can follow the [quickstart guide](https://github.com/triton-inference-server/triton_cli?tab=readme-ov-file#serving-a-trt-llm-model)
in the Triton CLI Github repository to serve GPT-2 on the Triton server with the TensorRT-LLM backend.

### Run GenAI-Perf

Run GenAI-Perf inside the Triton Inference Server SDK container:

```bash
genai-perf profile \
  -m gpt2 \
  --service-kind triton \
  --backend tensorrtllm \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --streaming
```

Example output:

```
                              NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│          Time to first token (ms) │  13.68 │  11.07 │  21.50 │  18.81 │  14.29 │  13.97 │
│          Inter token latency (ms) │   1.86 │   1.28 │   2.11 │   2.11 │   2.01 │   1.95 │
│              Request latency (ms) │ 203.70 │ 180.33 │ 228.30 │ 225.45 │ 216.48 │ 211.72 │
│            Output sequence length │ 103.46 │  95.00 │ 134.00 │ 122.96 │ 108.00 │ 104.75 │
│             Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
│ Output token throughput (per sec) │ 504.02 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request throughput (per sec) │   4.87 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

## Profile GPT-2 running on Triton + vLLM <a id="triton-vllm"></a>

You can follow the [quickstart guide](https://github.com/triton-inference-server/triton_cli?tab=readme-ov-file#serving-a-vllm-model)
in the Triton CLI Github repository to serve GPT-2 on the Triton server with the vLLM backend.

### Run GenAI-Perf

Run GenAI-Perf inside the Triton Inference Server SDK container:

```bash
genai-perf profile \
  -m gpt2 \
  --service-kind triton \
  --backend vllm \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --streaming
```

Example output:

```
                              NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│          Time to first token (ms) │  22.04 │  14.00 │  26.02 │  25.73 │  24.41 │  24.06 │
│          Inter token latency (ms) │   4.58 │   3.45 │   5.34 │   5.33 │   5.11 │   4.86 │
│              Request latency (ms) │ 542.48 │ 468.10 │ 622.39 │ 615.67 │ 584.73 │ 555.90 │
│            Output sequence length │ 115.15 │ 103.00 │ 143.00 │ 138.00 │ 120.00 │ 118.50 │
│             Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
│ Output token throughput (per sec) │ 212.04 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request throughput (per sec) │   1.84 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

## Profile Zephyr-7B-Beta running on OpenAI Chat API-Compatible Server <a id="openai-chat"></a>

Serve the model on the vLLM server with [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) endpoint:

```bash
docker run -it --net=host --gpus=all vllm/vllm-openai:latest --model HuggingFaceH4/zephyr-7b-beta --dtype float16
```

### Run GenAI-Perf

Run GenAI-Perf inside the Triton Inference Server SDK container:

```bash
genai-perf profile \
  -m HuggingFaceH4/zephyr-7b-beta \
  --service-kind openai \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --streaming \
  --tokenizer HuggingFaceH4/zephyr-7b-beta
```

Example output:

```
                                    NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│          Time to first token (ms) │    37.99 │    32.65 │    45.89 │    45.85 │    44.69 │    37.49 │
│          Inter token latency (ms) │    19.19 │    18.78 │    20.11 │    20.00 │    19.39 │    19.23 │
│              Request latency (ms) │ 1,915.41 │ 1,574.73 │ 2,027.20 │ 2,016.50 │ 1,961.22 │ 1,931.45 │
│            Output sequence length │    98.83 │    81.00 │   101.00 │   100.83 │   100.00 │   100.00 │
│             Input sequence length │   200.00 │   200.00 │   200.00 │   200.00 │   200.00 │   200.00 │
│ Output token throughput (per sec) │    51.55 │      N/A │      N/A │      N/A │      N/A │      N/A │
│      Request throughput (per sec) │     0.52 │      N/A │      N/A │      N/A │      N/A │      N/A │
└───────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## Profile GPT-2 running on OpenAI Completions API-Compatible Server <a id="openai-completions"></a>

Serve the model on the vLLM server with [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions) endpoint:

```bash
docker run -it --net=host --gpus=all vllm/vllm-openai:latest --model gpt2 --dtype float16 --max-model-len 1024
```

### Run GenAI-Perf

Run GenAI-Perf inside the Triton Inference Server SDK container:

```bash
genai-perf profile \
  -m gpt2 \
  --service-kind openai \
  --endpoint-type completions \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0
```

Example output:

```
                             NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│              Request latency (ms) │ 437.85 │ 328.10 │ 497.05 │ 495.28 │ 485.68 │ 460.91 │
│            Output sequence length │ 112.66 │  83.00 │ 123.00 │ 122.69 │ 119.90 │ 116.25 │
│             Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
│ Output token throughput (per sec) │ 257.21 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request throughput (per sec) │   2.28 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```
