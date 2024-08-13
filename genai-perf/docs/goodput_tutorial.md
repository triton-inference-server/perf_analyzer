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

# Tutorials

## Profile GPT2 Goodput running on Triton + vLLM <a id="triton-vllm"></a>

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
```

### Run GenAI-Perf with valid goodput constraints

```bash
# Valid goodput constraints
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
  --concurrency 2 \
  --measurement-interval 800 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --goodput time_to_first_tokens:8 inter_token_latencies:2 request_latencies:300
```

Example output:

```
                                   LLM Metrics                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   5.95 │   3.96 │  17.73 │  17.72 │   8.00 │   5.01 │
│ Inter token latency (ms) │   1.67 │   1.46 │   1.98 │   1.95 │   1.80 │   1.72 │
│     Request latency (ms) │ 191.52 │ 176.96 │ 212.54 │ 212.54 │ 210.86 │ 196.97 │
│   Output sequence length │ 112.27 │ 104.00 │ 124.00 │ 123.71 │ 119.00 │ 115.00 │
│    Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1171.61
Request throughput (per sec): 10.44
Request goodput (per sec): 9.39
```

### Run GenAI-Perf with invalid goodput constraints

```bash
# Invalid goodput constraints
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
  --concurrency 2 \
  --measurement-interval 800 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
  --goodput time_to_first_tokens:8 inter_token_latencies:2 output_token_throughputs_per_requesdt:650
```

Example output:

```
2024-08-12 17:03 [INFO] genai_perf.goodput_calculator.llm_goodput_calculator:90 - 
Invalid SLOs found: output_token_throughputs_per_requesdt. The goodput will be N/A. 
Valid SLOs are: time_to_first_token, inter_token_latency, request_latency, 
output_token_throughput_per_request in plural forms.
                                   LLM Metrics                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   5.52 │   3.91 │  21.72 │  21.62 │   5.94 │   4.98 │
│ Inter token latency (ms) │   1.53 │   1.29 │   1.79 │   1.78 │   1.59 │   1.56 │
│     Request latency (ms) │ 176.91 │ 170.80 │ 214.56 │ 214.49 │ 179.84 │ 176.55 │
│   Output sequence length │ 113.47 │ 107.00 │ 130.00 │ 127.21 │ 119.80 │ 116.00 │
│    Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1282.33
Request throughput (per sec): 11.30
Request goodput (per sec): N/A
```

### Run GenAI-Perf with no goodput constraints

```bash
# No goodput constraints
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
  --concurrency 2 \
  --measurement-interval 800 \
  --profile-export-file my_profile_export.json \
  --url localhost:8001 \
```

Example output:

```
                                   LLM Metrics                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   6.87 │   4.83 │  24.95 │  24.95 │   7.09 │   5.93 │
│ Inter token latency (ms) │   1.84 │   1.48 │   2.85 │   2.77 │   1.96 │   1.84 │
│     Request latency (ms) │ 210.63 │ 192.89 │ 329.89 │ 329.88 │ 210.44 │ 206.05 │
│   Output sequence length │ 112.19 │ 102.00 │ 140.00 │ 135.50 │ 119.00 │ 113.00 │
│    Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1065.01
Request throughput (per sec): 9.49
```

## Profile Embeddings Models Goodput with GenAI-Perf

## Create a Sample Embeddings Input File

To create a sample embeddings input file, use the following command:

```bash
echo '{"text": "What was the first car ever driven?"}
{"text": "Who served as the 5th President of the United States of America?"}
{"text": "Is the Sydney Opera House located in Australia?"}
{"text": "In what state did they film Shrek 2?"}' > embeddings.jsonl
```

This will generate a file named embeddings.jsonl with the following content:
```jsonl
{"text": "What was the first car ever driven?"}
{"text": "Who served as the 5th President of the United States of America?"}
{"text": "Is the Sydney Opera House located in Australia?"}
{"text": "In what state did they film Shrek 2?"}
```

## Start an OpenAI Embeddings-Compatible Server
To start an OpenAI embeddings-compatible server, run the following command:
```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model intfloat/e5-mistral-7b-instruct --dtype float16 --max-model-len 1024
```

## Run GenAI-Perf
To profile embeddings models using GenAI-Perf, use the following command:

### Run GenAI-Perf with no goodput constraints

```bash
# No goodput constraints
genai-perf profile \
    -m intfloat/e5-mistral-7b-instruct \
    --service-kind openai \
    --endpoint-type embeddings \
    --batch-size 2 \
    --input-file embeddings.jsonl \
    --measurement-interval 1000
```
Example output:

```
                           Embeddings Metrics                           
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃            Statistic ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Request latency (ms) │ 22.54 │ 22.02 │ 31.68 │ 23.74 │ 22.83 │ 22.58 │
└──────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘
Request throughput (per sec): 44.16
```

### Run GenAI-Perf with valid goodput constraints

```bash
# Valid goodput constraints
genai-perf profile \
    -m intfloat/e5-mistral-7b-instruct \
    --service-kind openai \
    --endpoint-type embeddings \
    --batch-size 2 \
    --input-file embeddings.jsonl \
    --measurement-interval 1000 \
    --goodput request_latencies:22.5
```
Example output:

```
                           Embeddings Metrics                           
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃            Statistic ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Request latency (ms) │ 22.57 │ 22.11 │ 31.78 │ 24.06 │ 22.86 │ 22.56 │
└──────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘
Request throughput (per sec): 44.14
Request goodput (per sec): 28.22
```

### Run GenAI-Perf with invalid goodput constraints

```bash
# Invalid goodput constraints
genai-perf profile \
    -m intfloat/e5-mistral-7b-instruct \
    --service-kind openai \
    --endpoint-type embeddings \
    --batch-size 2 \
    --input-file embeddings.jsonl \
    --measurement-interval 1000 \
    --goodput request_latencies:22.5 time_to_first_tokens:2
```
Example output:

```
2024-08-12 17:42 [INFO] genai_perf.goodput_calculator.llm_goodput_calculator:90 - Invalid SLOs found: time_to_first_tokens. The goodput will be N/A. Valid SLOs are: request_latency in plural forms.
                           Embeddings Metrics                           
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃            Statistic ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Request latency (ms) │ 22.49 │ 22.04 │ 31.82 │ 23.74 │ 22.82 │ 22.50 │
└──────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘
Request throughput (per sec): 44.35
Request goodput (per sec): N/A
```
