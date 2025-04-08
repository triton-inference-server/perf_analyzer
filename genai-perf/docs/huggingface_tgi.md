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

# Profile Hugging Face TGI Models Using GenAI-Perf

GenAI-Perf can profile LLMs running on a
[Hugging Face's Text Generation Inference (TGI) API-compatible](https://huggingface.co/docs/chat-ui/en/configuration/models/providers/tgi)
server using the generate API. This guide walks you through:

## Step 1: Start a Hugging Face TGI Server

To launch a Hugging Face TGI server, use the official `ghcr.io` image:

```bash
docker run --gpus all --rm -it \
  -p 8080:80 \
  -e MODEL_ID=llava-hf/llava-v1.6-mistral-7b-hf \
  ghcr.io/huggingface/text-generation-inference
```

---

## Approach 1. Profile Using Synthetic Inputs

Run with built-in synthetic prompts:

```bash
genai-perf profile \
  -m llava-hf/llava-v1.6-mistral-7b-hf \
  --service-kind openai \
  --endpoint-type huggingface_generate \
  --url localhost:8080 \
  --batch-size-image 1 \
  --image-width-mean 100 \
  --image-height-mean 100 \
  --synthetic-input-tokens-mean 10
```

---

### Approach 2: Bring Your Own Data (BYOD)

Instead of letting GenAI-Perf create the synthetic data,
you can also provide GenAI-Perf with your own data using
[`--input-file`](../README.md#--input-file-path) CLI option.
The file needs to be in JSONL format and should contain both the prompt and
the filepath to the image to send.

For instance, an example of input file would look something as following:
```bash
// input.jsonl
{"text": "What is in this image?", "image": "path/to/image1.png"}
{"text": "What is the color of the dog?", "image": "path/to/image2.jpeg"}
{"text": "Describe the scene in the picture.", "image": "path/to/image3.png"}
```

#### 2. Run GenAI-Perf

```bash
genai-perf profile \
  -m llava-hf/llava-v1.6-mistral-7b-hf \
  --service-kind openai \
  --endpoint-type huggingface_generate \
  --url localhost:8080 \
  --input-file input.jsonl
```

## Review the Output

Example output:

```
                                        NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃                            Statistic ┃       avg ┃      min ┃       max ┃       p99 ┃       p90 ┃       p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│                 Request Latency (ms) │  6,675.95 │   234.82 │ 54,474.75 │ 50,240.69 │ 12,134.14 │  1,283.24 │
│      Output Sequence Length (tokens) │    347.50 │     3.00 │  2,908.00 │  2,681.92 │    647.20 │     59.75 │
│       Input Sequence Length (tokens) │ 10,164.10 │ 7,293.00 │ 12,113.00 │ 12,113.00 │ 12,113.00 │ 12,112.75 │
│ Output Token Throughput (tokens/sec) │     52.05 │      N/A │       N/A │       N/A │       N/A │       N/A │
│         Request Throughput (per sec) │      0.15 │      N/A │       N/A │       N/A │       N/A │       N/A │
│                Request Count (count) │     10.00 │      N/A │       N/A │       N/A │       N/A │       N/A │
└──────────────────────────────────────┴───────────┴──────────┴───────────┴───────────┴───────────┴───────────┘
```