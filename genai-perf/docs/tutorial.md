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

- [Profile GPT2 running on Triton + TensorRT-LLM](#tensorrt-llm)
- [Profile GPT2 running on Triton + vLLM](#triton-vllm)
- [Profile GPT2 running on OpenAI Chat Completions API-Compatible Server](#openai-chat)
- [Profile GPT2 running on OpenAI Completions API-Compatible Server](#openai-completions)

---

## Profile GPT2 running on Triton + TensorRT-LLM <a id="tensorrt-llm"></a>

### Run GPT2 on Triton Inference Server using TensorRT-LLM

<details>
<summary>See instructions</summary>

Run Triton Inference Server with TensorRT-LLM backend container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

docker run -it --net=host --gpus=all --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:${RELEASE}-trtllm-python-py3

# Install Triton CLI (~5 min):
pip install "git+https://github.com/triton-inference-server/triton_cli@0.0.8"

# Download model:
triton import -m gpt2 --backend tensorrtllm

# Run server:
triton start
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from Triton Inference Server SDK container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

docker run -it --net=host --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Run GenAI-Perf in the container:
genai-perf profile \
  -m gpt2 \
  --service-kind triton \
  --backend tensorrtllm \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --streaming \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```

Example output:

```
                                        NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃                Statistic ┃         avg ┃         min ┃         max ┃         p99 ┃         p90 ┃         p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Time to first token (ns) │  13,266,974 │  11,818,732 │  18,351,779 │  16,513,479 │  13,741,986 │  13,544,376 │
│ Inter token latency (ns) │   2,069,766 │      42,023 │  15,307,799 │   3,256,375 │   3,020,580 │   2,090,930 │
│     Request latency (ns) │ 223,532,625 │ 219,123,330 │ 241,004,192 │ 238,198,306 │ 229,676,183 │ 224,715,918 │
│   Output sequence length │         104 │         100 │         129 │         128 │         109 │         105 │
│    Input sequence length │         199 │         199 │         199 │         199 │         199 │         199 │
└──────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
Output token throughput (per sec): 460.42
Request throughput (per sec): 4.44
```

## Profile GPT2 running on Triton + vLLM <a id="triton-vllm"></a>

### Run GPT2 on Triton Inference Server using vLLM

<details>
<summary>See instructions</summary>

Run Triton Inference Server with vLLM backend container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"


docker run -it --net=host --gpus=1 --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tritonserver:${RELEASE}-vllm-python-py3

# Install Triton CLI (~5 min):
pip install "git+https://github.com/triton-inference-server/triton_cli@0.0.8"

# Download model:
triton import -m gpt2 --backend vllm

# Run server:
triton start
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from Triton Inference Server SDK container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

docker run -it --net=host --gpus=1 nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Run GenAI-Perf in the container:
genai-perf profile \
  -m gpt2 \
  --service-kind triton \
  --backend vllm \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --streaming \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --output-tokens-mean-deterministic \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001
```

Example output:

```
                                     NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃                Statistic ┃         avg ┃         min ┃         max ┃         p99 ┃         p90 ┃         p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Time to first token (ns) │  15,786,560 │  11,437,189 │  49,550,549 │  40,129,652 │  21,248,091 │  17,824,695 │
│ Inter token latency (ns) │   3,543,380 │     591,898 │  10,013,690 │   6,152,260 │   5,039,278 │   4,060,982 │
│     Request latency (ns) │ 388,415,721 │ 312,552,612 │ 528,229,817 │ 518,189,390 │ 484,281,365 │ 459,417,637 │
│   Output sequence length │         113 │         105 │         123 │         122 │         119 │         115 │
│    Input sequence length │         199 │         199 │         199 │         199 │         199 │         199 │
└──────────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
Output token throughput (per sec): 290.24
Request throughput (per sec): 2.57
```

## Profile Zephyr running on OpenAI Chat API-Compatible Server <a id="openai-chat"></a>

### Run Zephyr on [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)-compatible server

<details>
<summary>See instructions</summary>

Run the vLLM inference server:

```bash
docker run -it --net=host --gpus=all vllm/vllm-openai:latest --model HuggingFaceH4/zephyr-7b-beta --dtype float16
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from Triton Inference Server SDK container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

docker run -it --net=host --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Run GenAI-Perf in the container:
genai-perf profile \
  -m HuggingFaceH4/zephyr-7b-beta \
  --service-kind openai \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --streaming \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
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

## Profile GPT2 running on OpenAI Completions API-Compatible Server <a id="openai-completions"></a>

### Running GPT2 on [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions)-compatible server

<details>
<summary>See instructions</summary>

Run the vLLM inference server:

```bash
docker run -it --net=host --gpus=all vllm/vllm-openai:latest --model gpt2 --dtype float16 --max-model-len 1024
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from Triton Inference Server SDK container:

```bash
export RELEASE="yy.mm" # e.g. export RELEASE="24.06"

docker run -it --net=host --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk


# Run GenAI-Perf in the container:
genai-perf profile \
  -m gpt2 \
  --service-kind openai \
  --endpoint v1/completions \
  --endpoint-type completions \
  --num-prompts 100 \
  --random-seed 123 \
  --synthetic-input-tokens-mean 200 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 100 \
  --output-tokens-stddev 0 \
  --tokenizer hf-internal-testing/llama-tokenizer \
  --concurrency 1 \
  --measurement-interval 4000 \
  --profile-export-file my_profile_export.json \
  --url localhost:8000
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
