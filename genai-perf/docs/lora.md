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

# Profile Multiple LoRA Adapters
GenAI-Perf allows you to profile multiple LoRA adapters on top of a base model.

## Select LoRA Adapters
To do this, list multiple adapters after the model name option `-m`:

```bash
genai-perf profile \
    -m lora_adapter1 lora_adapter2 lora_adapter3
```

## Choose a Strategy for Selecting Models
When profiling with multiple models, you can specify how the models should be
assigned to prompts using the `--model-selection-strategy` option:

```bash
genai-perf profile \
    -m lora_adapter1 lora_adapter2 lora_adapter3 \
    --model-selection-strategy round_robin
```

This setup will cycle through the lora_adapter1, lora_adapter2, and
lora_adapter3 models in a round-robin manner for each prompt.

For more details on additional options and configurations, refer to the
[Command Line Options section](../README.md#command-line-options) in the README.

## Profile Llama running on OpenAI Completions API-Compatible Server <a id="openai"></a>

### Run Llama on [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)-compatible server

<details>
<summary>See instructions</summary>

Download the adapters:

```bash
python3
from huggingface_hub import snapshot_download
lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
lora_path_2 = snapshot_download(repo_id="monsterapi/llama2-code-generation")
```
Run the vLLM inference server:

```bash
docker run -it --net=host --rm --gpus=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-hf \
    --dtype float16 \
    --max-model-len 1024 \
    --lora-modules \
    adapter1=/root/.cache/huggingface/hub/models--monsterapi--llama2-code-generation/snapshots/${SNAPSHOT_ID}/  \
    adapter2=/root/.cache/huggingface/hub/models--yard1-llama-2-7b-sql-lora-test/snapshots/${SNAPSHOT_ID}/ \
    --enable-lora
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from the Triton Inference Server SDK container:

```bash
export RELEASE="24.09"

docker run -it --net=host --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Run GenAI-Perf in the container:
genai-perf profile \
  -m adapter1 adapter2 \
  --service-kind openai \
  --endpoint-type completions \
  --model-selection-strategy round_robin
```

Example output:

```
                              NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│              Request latency (ms) │ 442.59 │ 175.95 │ 652.26 │ 608.05 │ 463.43 │ 449.82 │
│            Output sequence length │  16.84 │   2.00 │  19.00 │  19.00 │  17.00 │  17.00 │
│             Input sequence length │ 550.05 │ 550.00 │ 553.00 │ 551.40 │ 550.00 │ 550.00 │
│ Output token throughput (per sec) │  38.04 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request throughput (per sec) │   2.26 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

## Profile Mistral running on Hugging Face TGI Server <a id="tgi"></a>

### Run Mistral on [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/en/conceptual/lora) server

<details>
<summary>See instructions</summary>

Run the TGI  server:

```bash
mkdir data
model=mistralai/Mistral-7B-v0.1
volume=$PWD/data

docker run \
    --gpus all \
    --shm-size 1g \
    -p 8000:80 \
    -v $volume:/data \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    ghcr.io/huggingface/text-generation-inference:2.1.1 \
    --model-id $model \
    --lora-adapters=predibase/customer_support,predibase/magicoder
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from the Triton Inference Server SDK container:

```bash
export RELEASE="24.09"

docker run -it --net=host --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Run GenAI-Perf in the container:
genai-perf profile \
  -m predibase/customer_support predibase/magicoder \
  --service-kind openai \
  --endpoint-type completions \
  --model-selection-strategy round_robin
```

Example output:

```
                                   NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃                         Statistic ┃      avg ┃    min ┃      max ┃      p99 ┃      p90 ┃      p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│              Request latency (ms) │ 1,655.06 │ 155.95 │ 1,942.88 │ 1,941.54 │ 1,935.78 │ 1,927.85 │
│            Output sequence length │    88.43 │   6.00 │   108.00 │   107.80 │   106.00 │   103.00 │
│             Input sequence length │   550.00 │ 550.00 │   550.00 │   550.00 │   550.00 │   550.00 │
│ Output token throughput (per sec) │    53.43 │    N/A │      N/A │      N/A │      N/A │      N/A │
│      Request throughput (per sec) │     0.60 │    N/A │      N/A │      N/A │      N/A │      N/A │
└───────────────────────────────────┴──────────┴────────┴──────────┴──────────┴──────────┴──────────┘
```

## Profile Mistral running on Lorax Server <a id="lorax"></a>

### Run Mistral on [Lorax](https://github.com/predibase/lorax) server

<details>
<summary>See instructions</summary>

Run the TGI  server:

```bash
mkdir data
model=mistralai/Mistral-7B-Instruct-v0.1
volume=$PWD/data

docker run \
    --gpus all \
    --shm-size 1g \
    -p 8000:80 \
    -v $volume:/data \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
     ghcr.io/predibase/lorax:main \
    --model-id $model
```

</details>

### Run GenAI-Perf

Run GenAI-Perf from the Triton Inference Server SDK container:

```bash
export RELEASE="24.09"

docker run -it --net=host --gpus=all nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Run GenAI-Perf in the container:
genai-perf profile \
  -m alignment-handbook/zephyr-7b-dpo-lora Undi95/Mistral-7B-roleplay_alpaca-lora \
  --service-kind openai \
  --endpoint-type completions \
  --model-selection-strategy round_robin \
  --concurrency=128
```

Example output:

```
                                      NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃                         Statistic ┃       avg ┃      min ┃       max ┃       p99 ┃       p90 ┃       p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│              Request latency (ms) │ 20,277.80 │ 1,229.48 │ 33,166.99 │ 33,082.30 │ 32,320.14 │ 31,206.61 │
│            Output sequence length │     24.86 │     4.00 │     96.00 │     91.50 │     51.00 │     19.50 │
│             Input sequence length │    550.00 │   550.00 │    550.00 │    550.00 │    550.00 │    550.00 │
│ Output token throughput (per sec) │      5.25 │      N/A │       N/A │       N/A │       N/A │       N/A │
│      Request throughput (per sec) │      0.21 │      N/A │       N/A │       N/A │       N/A │       N/A │
└───────────────────────────────────┴───────────┴──────────┴───────────┴───────────┴───────────┴───────────┘
```