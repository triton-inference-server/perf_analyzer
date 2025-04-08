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

# Profile Multi-Modal Language Models with GenAI-Perf

GenAI-Perf allows you to profile Multi-Modal Language Models (MMLM) running on
[OpenAI Chat Completions API](https://platform.openai.com/docs/guides/chat-completions)-compatible server
by sending multi-modal contents to the server
(for instance, see OpenAI [vision](https://platform.openai.com/docs/guides/vision) and
[audio](https://platform.openai.com/docs/guides/audio?example=audio-in) inputs).


## Quickstart 1. Run GenAI-Perf on Vision Language Model (VLM)

Start OpenAI API compatible server with a VLM model using following command:

```bash
docker run --runtime nvidia --gpus all \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model llava-hf/llava-v1.6-mistral-7b-hf --dtype float16
```

Use GenAI-Perf to generate/send text and image request data to the server
```bash
genai-perf profile \
    -m llava-hf/llava-v1.6-mistral-7b-hf \
    --endpoint-type multimodal \
    --image-width-mean 50 \
    --image-height-mean 50 \
    --synthetic-input-tokens-mean 10 \
    --output-tokens-mean 10 \
    --streaming
```

Console output will have the following result table

```bash
                           NVIDIA GenAI-Perf | Multi-Modal Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃      max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│          Time To First Token (ms) │ 205.38 │ 172.31 │ 1,020.58 │ 246.02 │ 207.33 │ 204.96 │
│         Time To Second Token (ms) │  18.58 │  18.03 │    19.22 │  19.13 │  18.72 │  18.65 │
│              Request Latency (ms) │ 369.06 │ 336.56 │ 1,183.65 │ 408.84 │ 370.97 │ 368.50 │
│          Inter Token Latency (ms) │  16.41 │  16.27 │    18.32 │  18.16 │  16.47 │  16.41 │
│   Output Sequence Length (tokens) │  10.98 │  10.00 │    11.00 │  11.00 │  11.00 │  11.00 │
│    Input Sequence Length (tokens) │  10.06 │  10.00 │    11.00 │  11.00 │  10.00 │  10.00 │
│ Output Token Throughput (per sec) │  29.74 │    N/A │      N/A │    N/A │    N/A │    N/A │
│      Request Throughput (per sec) │   2.71 │    N/A │      N/A │    N/A │    N/A │    N/A │
│             Request Count (count) │  97.00 │    N/A │      N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴──────────┴────────┴────────┴────────┘
```

## Quickstart 2. Run GenAI-Perf on Multi-Modal Language Model (MMLM)

In this example, we will measure performance of the recent Multi-Modal Language Model (MMLM)
`Phi-4-multimodal-instruct` from Microsoft hosted on NVIDIA API which is OpenAI API compatible.
First, visit https://build.nvidia.com/microsoft/phi-4-multimodal-instruct and create API key.

Run GenAI-Perf to generate/send all three modalities to the server
```bash
export NVIDIA_API_KEY=your_api_key

genai-perf profile \
    -m microsoft/phi-4-multimodal-instruct \
    -u https://integrate.api.nvidia.com \
    --endpoint-type multimodal \
    --synthetic-input-tokens-mean 10 \
    --output-tokens-mean 10 \
    --image-width-mean 50 \
    --image-height-mean 50 \
    --audio-length-mean 3 \
    --audio-depths 16 32 \
    --audio-sample-rates 16 44.1 48 \
    --audio-num-channels 2 \
    --audio-format wav \
    --streaming \
    --header "Authorization: Bearer '${NVIDIA_API_KEY}'"
```

Console output will have the following result table

```bash
                          NVIDIA GenAI-Perf | Multi-Modal Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│          Time To First Token (ms) │ 284.94 │ 228.38 │ 562.57 │ 371.15 │ 325.76 │ 295.26 │
│         Time To Second Token (ms) │  10.02 │   8.79 │  10.94 │  10.89 │  10.59 │  10.32 │
│              Request Latency (ms) │ 346.80 │ 251.96 │ 662.96 │ 463.03 │ 404.79 │ 382.18 │
│          Inter Token Latency (ms) │   9.56 │   0.00 │  15.76 │  14.22 │  11.05 │  10.81 │
│   Output Sequence Length (tokens) │   7.43 │   1.00 │  14.00 │  14.00 │  12.00 │  10.00 │
│    Input Sequence Length (tokens) │  10.11 │  10.00 │  11.00 │  11.00 │  11.00 │  10.00 │
│ Output Token Throughput (per sec) │  20.96 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request Throughput (per sec) │   2.82 │    N/A │    N/A │    N/A │    N/A │    N/A │
│             Request Count (count) │ 101.00 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```


## Generating Multi-Modal Requests with GenAI-Perf

Currently, you can send multi-modal contents with GenAI-Perf using the following two approaches:
1. The synthetic data generation approach, where GenAI-Perf generates the multi-modal data for you.
2. The Bring Your Own Data (BYOD) approach, where you provide GenAI-Perf with the data to send.


### Approach 1: Synthetic Multi-Modal Data Generation

GenAI-Perf can generate synthetic data of three modalities (text, image, and audio)
using the modality-specific parameters provide by the user through CLI.
Checkout [CLI Input Options](../README.md#input-options) for a complete list of parameters
that you can tweak for different modalities.

```bash
genai-perf profile \
    -m <multimodal_model> \
    --endpoint-type multimodal \
    # audio parameters
    --audio-length-mean 10 \
    --audio-length-stddev 2 \
    --audio-depths 16 32 \
    --audio-sample-rates 16 44.1 48 \
    --audio-num-channels 1 \
    --audio-format wav \
    # image parameters
    --image-width-mean 512 \
    --image-width-stddev 30 \
    --image-height-mean 512 \
    --image-height-stddev 30 \
    --image-format png \
    # text parameters
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --streaming
```

> [!Note]
> Under the hood, GenAI-Perf generates synthetic images using a few source images
> under the `inputs/source_images` directory.
> If you would like to add/remove/edit the source images,
> you can do so by directly editing the source images under the directory.
> GenAI-Perf will pickup the images under the directory automatically when
> generating the synthetic images.


### Approach 2: Bring Your Own Data (BYOD)

> [!Note]
> This approach only supports text and image inputs at the moment.

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
...
```

After you create the file, you can run GenAI-Perf using the following command:

```bash
genai-perf profile \
    -m <multimodal_model> \
    --endpoint-type multimodal \
    --input-file input.jsonl \
    --streaming
```
