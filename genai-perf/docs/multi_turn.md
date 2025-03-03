<!--
Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Benchmark Multi-Turn Chat with GenAI-Perf

GenAI-Perf allows you to benchmark multi-turn chat. This can be used for
simulating multiple turns in a conversation in a way that matches real-world
user behavior.

You can use either synthetic data or a custom dataset.
This tutorial will guide you through setting up a model server and running a
profiling session with simulated conversations or a predefined dataset.

## Table of Contents

- [Start a Chat Model Server](#start-a-chat-model-server)
- [Approach 1: Benchmark with Synthetic Data](#approach-1-benchmark-with-synthetic-data)
- [Approach 2: Benchmark with a Custom Dataset](#approach-2-benchmark-with-a-custom-dataset)
- [Review the Output](#review-the-output)

## Start a Chat Model Server

First, launch a vLLM server with an chat endpoint:

```bash
docker run -it --net=host --rm --gpus=all \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float16 \
  --max-model-len 1024
```

## Approach 1: Benchmark with Synthetic Data

Use synthetic data to simulate multiple chat sessions with controlled token
input and response lengths.

```bash
genai-perf profile \
  -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --service-kind openai \
  --endpoint-type chat \
  --num-sessions 10 \
  --session-concurrency 5 \
  --session-turns-mean 2 \
  --session-turns-stddev 0 \
  --session-turn-delay-mean 1000 \
  --session-turn-delay-stddev 5 \
  --synthetic-input-tokens-mean 50 \
  --output-tokens-mean 50 \
  --num-prefix-prompts 3 \
  --prefix-prompt-length 15
```

### Understand Key Arguments

#### Required Arguments
- `--num-sessions 10`: Simulates 10 independent chat sessions.
- `--session-concurrency 5`: Enables session mode and runs up to 5 sessions in
parallel.

#### Optional Arguments
- `--session-turns-mean 2`: Each session has an average of 2 turns.
- `--session-turn-delay-mean 1000`: Introduces a 1-second delay between user
turns (simulating real-world interaction).
- `--synthetic-input-tokens-mean 50`: Each user input averages 50 tokens.
- `--output-tokens-mean 50`: Each model response averages 50 tokens.
- `--num-prefix-prompts 3`: Uses a pool of 3 system prompts for the first
turn in each session.
- `--prefix-prompt-length 15`: Each prefix prompt contains 15 tokens.

---

## Approach 2: Benchmark with a Custom Dataset

If you prefer to benchmark using a predefined dataset, create a JSONL input file
with the dataset.

### Example Input File

```bash
echo '{"session_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6", "delay": 1, "input_length": 50, "output_length": 10}
{"session_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6", "delay": 2, "input_length": 50, "output_length": 10}
{"session_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6", "input_length": 100, "output_length": 10}
{"session_id": "113059749145936325402354257176981405696", "delay": 0, "input_length": 25, "output_length": 20}
{"session_id": "113059749145936325402354257176981405696", "input_length": 20, "output_length": 20}' > inputs.jsonl
```

### Understand Key Fields

#### Required Fields
- `delay`: Sets the delay to wait after receiving a response before
sending the next request. This field is required except for the
last turn in a session.

#### Optional Fields
- `input_length`: Sets the token length of the input for this request.
- `output_length`: Sets the token length of the output for this request.
- `text`: Provides the prompt text, if you prefer to bring your own rather
than have it be synthetically generated.

### Run GenAI-Perf with Custom Input

```bash
genai-perf profile \
  -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --service-kind openai \
  --endpoint-type chat \
  --input-file payload:inputs.jsonl \
  --session-concurrency 2
```

## Review the Output

Example output:

```
                             NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃                         Statistic ┃    avg ┃   min ┃    max ┃    p99 ┃    p90 ┃   p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│              Request Latency (ms) │  80.88 │ 50.25 │ 124.29 │ 123.21 │ 113.50 │ 97.31 │
│   Output Sequence Length (tokens) │  14.00 │ 10.00 │  20.00 │  20.00 │  20.00 │ 20.00 │
│    Input Sequence Length (tokens) │  38.80 │ 22.00 │  50.00 │  50.00 │  50.00 │ 50.00 │
│ Output Token Throughput (per sec) │ 315.70 │   N/A │    N/A │    N/A │    N/A │   N/A │
│      Request Throughput (per sec) │  22.55 │   N/A │    N/A │    N/A │    N/A │   N/A │
│             Request Count (count) │   5.00 │   N/A │    N/A │    N/A │    N/A │   N/A │
└───────────────────────────────────┴────────┴───────┴────────┴────────┴────────┴───────┘
```
