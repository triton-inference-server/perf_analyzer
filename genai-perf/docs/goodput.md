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

# Benchmark Goodput with GenAI-Perf

## Context

Goodput is defined as the number of completed requests per second
that meet specified metric constraints, also called service level
objectives.

For example, perhaps you want to measure the user experience of your service
by considering throughput only including requests where the time to first token
is under 50ms and inter-token latency is under 10ms.

GenAI-Perf provides this value as goodput.

## Tutorials

Below you can find tutorials on how to benchmark different models
using goodput.

### Examples

- [LLM Examples](#LLM)

- [Embedding Model Examples](#embeddings)

### Profile LLM Goodput<a id="LLM"></a>

#### Run GPT2 on an OpenAI Chat Completions API-compatible server

```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model gpt2 --dtype float16 --max-model-len 1024
```

#### Run GenAI-Perf with Goodput Constraints

```bash
genai-perf profile \
    -m gpt2 \
    --service-kind openai \
    --endpoint-type chat \
    --measurement-interval 1000 \
    --streaming \
    --goodput time_to_first_token:4.35 inter_token_latency:1.1
```

Example output:

```
                                   LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   4.44 │   3.63 │  23.85 │  13.30 │   5.15 │   4.20 │
│ Inter token latency (ms) │   1.03 │   0.76 │   1.92 │   1.63 │   1.18 │   1.08 │
│     Request latency (ms) │  22.31 │   7.67 │  45.83 │  41.11 │  25.18 │  21.97 │
│   Output sequence length │  18.54 │   5.00 │  24.00 │  22.41 │  21.00 │  20.00 │
│    Input sequence length │ 550.06 │ 550.00 │ 553.00 │ 551.82 │ 550.00 │ 550.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 827.09
Request throughput (per sec): 44.62
Request goodput (per sec): 30.95
```

### Profile Embeddings Models Goodput<a id="embeddings"></a>

#### Create a Sample Embeddings Input File

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

#### Start an OpenAI Embeddings-Compatible Server

To start an OpenAI embeddings-compatible server, run the following command:
```bash
docker run -it --net=host --rm --gpus=all vllm/vllm-openai:latest --model intfloat/e5-mistral-7b-instruct --dtype float16 --max-model-len 1024
```

#### Run GenAI-Perf with goodput constraints

```bash
genai-perf profile \
    -m intfloat/e5-mistral-7b-instruct \
    --service-kind openai \
    --endpoint-type embeddings \
    --batch-size 2 \
    --input-file embeddings.jsonl \
    --measurement-interval 1000 \
    --goodput request_latency:22.5
```
Example output:

```
                           Embeddings Metrics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃            Statistic ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Request latency (ms) │ 22.23 │ 21.67 │ 31.96 │ 22.90 │ 22.48 │ 22.31 │
└──────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘
Request throughput (per sec): 44.73
Request goodput (per sec): 40.28
```