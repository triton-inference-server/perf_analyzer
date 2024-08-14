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

# Context

Goodput, defined as the number of completed requests per second that meet the Service Level Objectives (SLOs), provides an enhanced measure of AI serving performance by accounting for both cost efficiency and user satisfaction.

GenAI-Perf is continually evolving to meet the demands of next-generation GenAI workflows. We are excited to introduce our latest feature: goodput support for measuring Service Level Objectives.

This new feature empowers you to more precisely evaluate the efficiency and effectiveness of your AI models and services. By measuring goodput, GenAI-Perf delivers deeper insights into how well your services adhere to your defined SLOs, enabling you to optimize both performance and cost efficiency while prioritizing the user experience.

# Tutorials

This is a tutorial on how to use our goodput feature by examples.
## Examples

- [LLM Examples](#LLM)

- [Embedding Model Examples](#embeddings)

- [Ranking Model Examples](#rankings)

- [VLM Examples](#VLM)

## Profile GPT2 Goodput running on Triton + vLLM <a id="LLM"></a>

### Run GPT2 on Triton Inference Server using vLLM

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

### Run GenAI-Perf with valid goodput constraints

```bash
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
  --goodput time_to_first_token:8 inter_token_latency:2 request_latency:300
```

Example output:

```
                                   LLM Metrics                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   5.50 │   3.78 │  19.25 │  19.20 │   5.86 │   4.98 │
│ Inter token latency (ms) │   1.52 │   1.35 │   1.67 │   1.66 │   1.62 │   1.58 │
│     Request latency (ms) │ 176.02 │ 165.97 │ 197.03 │ 196.99 │ 181.80 │ 177.86 │
│   Output sequence length │ 113.34 │ 107.00 │ 127.00 │ 125.14 │ 118.00 │ 116.00 │
│    Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1287.08
Request throughput (per sec): 11.36
Request goodput (per sec): 10.65
```

### Run GenAI-Perf with invalid goodput constraints

```bash
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
2024-08-14 15:18 [INFO] genai_perf.goodput_calculator.llm_goodput_calculator:92 - Invalid SLOs found: time_to_first_tokens, inter_token_latencies, output_token_throughputs_per_requesdt. The goodput will be -1. Valid SLOs are: time_to_first_token, inter_token_latency, request_latency, output_token_throughput_per_request.
                                   LLM Metrics                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ Time to first token (ms) │   5.19 │   3.88 │  16.79 │  16.77 │   5.94 │   4.96 │
│ Inter token latency (ms) │   1.56 │   1.44 │   1.69 │   1.68 │   1.63 │   1.60 │
│     Request latency (ms) │ 176.93 │ 170.95 │ 197.70 │ 197.68 │ 177.86 │ 176.21 │
│   Output sequence length │ 111.47 │ 103.00 │ 123.00 │ 122.38 │ 118.80 │ 114.00 │
│    Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1259.65
Request throughput (per sec): 11.30
Request goodput (per sec): -1.00
```

### Run GenAI-Perf with no goodput constraints

```bash
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
│ Time to first token (ms) │   6.24 │   4.82 │  21.70 │  21.69 │   6.24 │   5.00 │
│ Inter token latency (ms) │   1.70 │   1.47 │   1.85 │   1.83 │   1.79 │   1.76 │
│     Request latency (ms) │ 195.38 │ 183.89 │ 215.47 │ 215.46 │ 198.31 │ 197.22 │
│   Output sequence length │ 112.46 │ 104.00 │ 131.00 │ 128.84 │ 119.30 │ 114.00 │
│    Input sequence length │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │ 200.00 │
└──────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
Output token throughput (per sec): 1150.71
Request throughput (per sec): 10.23
```

## Profile Embeddings Models Goodput with GenAI-Perf <a id="embeddings"></a>

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
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃            Statistic ┃   avg ┃   min ┃    max ┃   p99 ┃   p90 ┃   p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Request latency (ms) │ 23.51 │ 21.84 │ 200.79 │ 23.17 │ 22.70 │ 22.47 │
└──────────────────────┴───────┴───────┴────────┴───────┴───────┴───────┘
Request throughput (per sec): 42.39
```

### Run GenAI-Perf with valid goodput constraints

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

### Run GenAI-Perf with invalid goodput constraints

```bash
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
2024-08-14 15:24 [INFO] genai_perf.goodput_calculator.llm_goodput_calculator:92 - Invalid SLOs found: request_latencies, time_to_first_tokens. The goodput will be -1. Valid SLOs are: request_latency.
                           Embeddings Metrics                           
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃            Statistic ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ Request latency (ms) │ 22.38 │ 21.89 │ 32.48 │ 22.94 │ 22.61 │ 22.46 │
└──────────────────────┴───────┴───────┴───────┴───────┴───────┴───────┘
Request throughput (per sec): 44.50
Request goodput (per sec): -1.00
```

## Profile Ranking Models Goodput with GenAI-Perf <a id="rankings"></a>

GenAI-Perf allows you to profile ranking models compatible with Hugging Face's
[Text Embeddings Inference's re-ranker API](https://huggingface.co/docs/text-embeddings-inference/en/quick_tour#re-rankers).

## Create a Sample Rankings Input Directory

To create a sample rankings input directory, follow these steps:

Create a directory called rankings_jsonl:
```bash
mkdir rankings_jsonl
```

Inside this directory, create a JSONL file named queries.jsonl with queries data:

```bash
echo '{"text": "What was the first car ever driven?"}
{"text": "Who served as the 5th President of the United States of America?"}
{"text": "Is the Sydney Opera House located in Australia?"}
{"text": "In what state did they film Shrek 2?"}' > rankings_jsonl/queries.jsonl
```

Create another JSONL file named passages.jsonl with passages data:

```bash
echo '{"text": "Eric Anderson (born January 18, 1968) is an American sociologist and sexologist."}
{"text": "Kevin Loader is a British film and television producer."}
{"text": "Francisco Antonio Zea Juan Francisco Antonio Hilari was a Colombian journalist, botanist, diplomat, politician, and statesman who served as the 1st Vice President of Colombia."}
{"text": "Daddys Home 2 Principal photography on the film began in Massachusetts in March 2017 and it was released in the United States by Paramount Pictures on November 10, 2017. Although the film received unfavorable reviews, it has grossed over $180 million worldwide on a $69 million budget."}' > rankings_jsonl/passages.jsonl
```

## Start a Hugging Face Re-Ranker-Compatible Server
To start a Hugging Face re-ranker-compatible server, run the following commands:

```bash
model=BAAI/bge-reranker-large
revision=refs/pr/4
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.3 --model-id $model --revision $revision
```
## Run GenAI-Perf

### Run GenAI-Perf with no goodput constraints
To profile ranking models using GenAI-Perf, use the following command:

```bash
genai-perf profile \
    -m BAAI/bge-reranker-large \
    --service-kind openai \
    --endpoint-type rankings \
    --endpoint rerank \
    --input-file rankings_jsonl/ \
    -u localhost:8080 \
    --extra-inputs rankings:tei \
    --batch-size 2 \
    --measurement-interval 1000
```

This command specifies the use of Hugging Face's ranking API with `--endpoint rerank` and `--extra-inputs rankings:tei`.

Example output:

```
                         Rankings Metrics                          
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓
┃            Statistic ┃  avg ┃  min ┃   max ┃  p99 ┃  p90 ┃  p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩
│ Request latency (ms) │ 3.20 │ 1.66 │ 21.72 │ 4.29 │ 3.49 │ 3.40 │
└──────────────────────┴──────┴──────┴───────┴──────┴──────┴──────┘
Request throughput (per sec): 308.70
```

### Run GenAI-Perf with valid goodput constraints
To profile ranking models using GenAI-Perf, use the following command:

```bash
genai-perf profile \
    -m BAAI/bge-reranker-large \
    --service-kind openai \
    --endpoint-type rankings \
    --endpoint rerank \
    --input-file rankings_jsonl/ \
    -u localhost:8080 \
    --extra-inputs rankings:tei \
    --batch-size 2 \
    --measurement-interval 1000 \
    --goodput request_latency:3.3
```

This command specifies the use of Hugging Face's ranking API with `--endpoint rerank` and `--extra-inputs rankings:tei`.

Example output:

```
                         Rankings Metrics                          
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓
┃            Statistic ┃  avg ┃  min ┃   max ┃  p99 ┃  p90 ┃  p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩
│ Request latency (ms) │ 3.17 │ 1.65 │ 11.67 │ 3.70 │ 3.46 │ 3.39 │
└──────────────────────┴──────┴──────┴───────┴──────┴──────┴──────┘
Request throughput (per sec): 313.41
Request goodput (per sec): 128.37
```

### Run GenAI-Perf with invalid goodput constraints
To profile ranking models using GenAI-Perf, use the following command:

```bash
genai-perf profile \
    -m BAAI/bge-reranker-large \
    --service-kind openai \
    --endpoint-type rankings \
    --endpoint rerank \
    --input-file rankings_jsonl/ \
    -u localhost:8080 \
    --extra-inputs rankings:tei \
    --batch-size 2 \
    --measurement-interval 1000 \
    --goodput request_latencies:3.3 inter_token_latencies:0.2
```

This command specifies the use of Hugging Face's ranking API with `--endpoint rerank` and `--extra-inputs rankings:tei`.

Example output:

```
2024-08-14 15:26 [INFO] genai_perf.goodput_calculator.llm_goodput_calculator:92 - Invalid SLOs found: request_latencies, inter_token_latencies. The goodput will be -1. Valid SLOs are: request_latency.
                         Rankings Metrics                          
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓
┃            Statistic ┃  avg ┃  min ┃   max ┃  p99 ┃  p90 ┃  p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩
│ Request latency (ms) │ 3.16 │ 1.64 │ 13.15 │ 3.84 │ 3.48 │ 3.40 │
└──────────────────────┴──────┴──────┴───────┴──────┴──────┴──────┘
Request throughput (per sec): 313.07
Request goodput (per sec): -1.00
```

## Profile Vision-Language Models Goodput with GenAI-Perf <a id="VLM"></a>

GenAI-Perf allows you to profile Vision-Language Models (VLM) running on
[OpenAI Chat Completions API](https://platform.openai.com/docs/guides/chat-completions)-compatible server
by sending [multi-modal content](https://platform.openai.com/docs/guides/vision) to the server.

You can start OpenAI API compatible server with a VLM model using following command:

```bash
docker run --runtime nvidia --gpus all \
    --pull=always \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model llava-hf/llava-v1.6-mistral-7b-hf --dtype float16
```

### Run GenAI-Perf with no goodput constraints

```bash
genai-perf profile \
    -m llava-hf/llava-v1.6-mistral-7b-hf \
    --service-kind openai \
    --endpoint-type vision \
    --image-width-mean 512 \
    --image-width-stddev 30 \
    --image-height-mean 512 \
    --image-height-stddev 30 \
    --image-format png \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --measurement-interval 8000 \
    --streaming
```

Example Output

```
                                         LLM Metrics                                          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to first token (ms) │   363.85 │   312.47 │   556.26 │   537.84 │   372.08 │   361.68 │
│ Inter token latency (ms) │    17.66 │    16.96 │    18.56 │    18.51 │    18.11 │    17.87 │
│     Request latency (ms) │ 2,516.98 │ 1,407.99 │ 3,826.30 │ 3,739.81 │ 2,961.44 │ 2,935.26 │
│   Output sequence length │   123.00 │    58.00 │   193.00 │   188.70 │   150.00 │   148.00 │
│    Input sequence length │   100.00 │   100.00 │   100.00 │   100.00 │   100.00 │   100.00 │
└──────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
Output token throughput (per sec): 48.55
Request throughput (per sec): 0.39
```
### Run GenAI-Perf with valid goodput constraints

```bash
genai-perf profile \
    -m llava-hf/llava-v1.6-mistral-7b-hf \
    --service-kind openai \
    --endpoint-type vision \
    --image-width-mean 512 \
    --image-width-stddev 30 \
    --image-height-mean 512 \
    --image-height-stddev 30 \
    --image-format png \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --measurement-interval 8000 \
    --goodput time_to_first_token:350 inter_token_latency:17.5 request_latency:3200 \
    --streaming
```

Example Output

```
                                        LLM Metrics                                         
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                Statistic ┃      avg ┃    min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to first token (ms) │   356.96 │ 323.30 │   386.87 │   386.30 │   381.25 │   371.97 │
│ Inter token latency (ms) │    17.58 │  16.45 │    18.30 │    18.27 │    18.01 │    17.91 │
│     Request latency (ms) │ 2,525.10 │ 752.85 │ 4,047.83 │ 3,990.77 │ 3,477.18 │ 3,037.24 │
│   Output sequence length │   124.27 │  21.00 │   209.00 │   205.60 │   175.00 │   151.00 │
│    Input sequence length │   100.00 │ 100.00 │   100.00 │   100.00 │   100.00 │   100.00 │
└──────────────────────────┴──────────┴────────┴──────────┴──────────┴──────────┴──────────┘
Output token throughput (per sec): 48.89
Request throughput (per sec): 0.39
Request goodput (per sec): 0.04
```
### Run GenAI-Perf with invalid goodput constraints

```bash
genai-perf profile \
    -m llava-hf/llava-v1.6-mistral-7b-hf \
    --service-kind openai \
    --endpoint-type vision \
    --image-width-mean 512 \
    --image-width-stddev 30 \
    --image-height-mean 512 \
    --image-height-stddev 30 \
    --image-format png \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --measurement-interval 8000 \
    --goodput time_to_first_tokens:400 inter_token_latencies:50 request_latencies:5000 output_token_throughputs_per_requestd:6 \
    --streaming
```

Example Output

```
2024-08-14 15:34 [INFO] genai_perf.goodput_calculator.llm_goodput_calculator:92 - Invalid SLOs found: time_to_first_tokens, inter_token_latencies, request_latencies, output_token_throughputs_per_requestd. The goodput will be -1. Valid SLOs are: time_to_first_token, inter_token_latency, request_latency, output_token_throughput_per_request.
                                         LLM Metrics                                          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                Statistic ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ Time to first token (ms) │   367.03 │   340.40 │   403.55 │   402.81 │   396.20 │   380.03 │
│ Inter token latency (ms) │    17.32 │    16.42 │    18.24 │    18.20 │    17.88 │    17.57 │
│     Request latency (ms) │ 2,728.97 │ 1,325.37 │ 4,864.16 │ 4,798.33 │ 4,205.89 │ 2,896.01 │
│   Output sequence length │   137.40 │    55.00 │   260.00 │   256.13 │   221.30 │   145.00 │
│    Input sequence length │   100.00 │   100.00 │   100.00 │   100.00 │   100.00 │   100.00 │
└──────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
Output token throughput (per sec): 50.04
Request throughput (per sec): 0.36
Request goodput (per sec): -1.00
```

