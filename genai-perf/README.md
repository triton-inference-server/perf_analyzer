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

# GenAI-Perf

GenAI-Perf is a command line tool for measuring the throughput and latency of
generative AI models as served through an inference server.
For large language models (LLMs), GenAI-Perf provides metrics such as
[output token throughput](#output_token_throughput_metric),
[time to first token](#time_to_first_token_metric),
[time to second token](#time_to_second_token_metric),
[inter token latency](#inter_token_latency_metric), and
[request throughput](#request_throughput_metric).
For a full list of metrics please see the [Metrics section](#metrics).

Users specify a model name, an inference server URL, the type of inputs to use
(synthetic or from a dataset defined via a file), and the type of load to generate
(number of concurrent requests, request rate).

GenAI-Perf generates the specified load, measures the performance of the
inference server and reports the metrics in a simple table as console output.
The tool also logs all results in a csv and json file that can be used to derive
additional metrics and visualizations. The inference server must already be
running when GenAI-Perf is run.

You can use GenAI-Perf to run performance benchmarks on
- [Large Language Models](docs/tutorial.md)
- [Multi-Modal Language Models](docs/multi_modal.md)
- [Embedding Models](docs/embeddings.md)
- [Ranking Models](docs/rankings.md)
- [Multiple LoRA Adapters](docs/lora.md)

You can also use GenAI-Perf to run benchmarks on your
custom APIs using either
[customizable frontends](docs/customizable_frontends.md)
or
[customizable payloads](docs/customizable_payloads.md).
Customizable frontends provide more customizability,
while customizable payloads allow you to specify
specific payload schemas using a Jinja2 template.

> [!Note]
> GenAI-Perf is currently in early release and under rapid development. While we
> will try to remain consistent, command line options and functionality are
> subject to change as the tool matures.

</br>

<!--
======================
INSTALLATION
======================
-->

## Installation

The easiest way to install GenAI-Perf is through pip.
### Install GenAI-Perf (Ubuntu 24.04, Python 3.10+)

```bash
pip install genai-perf
```
**NOTE**: you must already have CUDA 12 installed


<details>

<summary>Alternatively, to install the container:</summary>

[Triton Server SDK container](https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver)

Pull the latest release using the following command:

```bash
export RELEASE="25.01"

docker run -it --net=host --gpus=all  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

# Validate the genai-perf command works inside the container:
genai-perf --help
```

You can also build Perf Analyzer [from source](../docs/install.md#build-from-source) to use alongside GenAI-Perf as well.

</details>

</br>

<!--
======================
QUICK START
======================
-->

## Quick Start

In this quick start, we will use GenAI-Perf to run performance benchmarking on
the GPT-2 model running on Triton Inference Server with a TensorRT-LLM engine.

### Serve GPT-2 TensorRT-LLM model using Triton CLI

You can follow the [quickstart guide](https://github.com/triton-inference-server/triton_cli?tab=readme-ov-file#serving-a-trt-llm-model)
in the Triton CLI Github repository to serve GPT-2 on the Triton server with the TensorRT-LLM backend.
**NOTE**: pip dependency error messages can be safely ignored.

The full instructions are copied below for convenience:

```bash
# This container comes with all of the dependencies for building TRT-LLM engines
# and serving the engine with Triton Inference Server.
docker run -ti \
    --gpus all \
    --network=host \
    --shm-size=1g --ulimit memlock=-1 \
    -v /tmp:/tmp \
    -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

# Install the Triton CLI
pip install git+https://github.com/triton-inference-server/triton_cli.git@0.1.2

# Build TRT LLM engine and generate a Triton model repository pointing at it
triton remove -m all
triton import -m gpt2 --backend tensorrtllm

# Start Triton pointing at the default model repository
triton start
```

### Running GenAI-Perf

Now we can run GenAI-Perf inside the Triton Inference Server SDK container:

```bash
genai-perf profile \
  -m gpt2 \
  --backend tensorrtllm \
  --streaming
```

Example output:

```
                              NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│          Time to first token (ms) │  16.26 │  12.39 │  17.25 │  17.09 │  16.68 │  16.56 │
│          Inter token latency (ms) │   1.85 │   1.55 │   2.04 │   2.02 │   1.97 │   1.92 │
│              Request latency (ms) │ 499.20 │ 451.01 │ 554.61 │ 548.69 │ 526.13 │ 514.19 │
│            Output sequence length │ 261.90 │ 256.00 │ 298.00 │ 296.60 │ 270.00 │ 265.00 │
│             Input sequence length │ 550.06 │ 550.00 │ 553.00 │ 551.60 │ 550.00 │ 550.00 │
│ Output token throughput (per sec) │ 520.87 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request throughput (per sec) │   1.99 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

See [Tutorial](docs/tutorial.md) for additional examples.

</br>

<!--
=====================
Config File
====================
-->
## Configuration File
In addition to setting options via the command-line, GenAI-Perf supports the passing in of a config file (in YAML format). The command is:</br>
```bash
genai-perf config -f <config_file>
```

### Creating a Template Config File
In order to make it easier for you to use config files, we have added a new subcommand that generates a template config file containing all possible options, pre-populated to their default settings. The command to create this is:</br>
```bash
genai-perf create-template
```
By default the config file is named `genai_perf_config.yaml`, but you can change that by passing in a custom name using the `-f` option.</br>
For less experienced users, you can include `-v/--verbose` and the config file will also contain descriptions for each option (similar to what you would see using `-h` from the command line).

Here is a sample section of what the template config file looks like:
```
  endpoint:
    model_selection_strategy: round_robin
    backend: tensorrtllm
    custom:
    type: kserve
    streaming: False
    server_metrics_urls: http://localhost:8002/metrics
    url: localhost:8001
    grpc_method:
```
and with `--verbose`:
```bash
 endpoint:
    # When multiple model are specified, this is how a specific model should be assigned to a prompt.
    # round_robin: nth prompt in the list gets assigned to n-mod len(models).
    # random: assignment is uniformly random
    model_selection_strategy: round_robin

    # When benchmarking Triton, this is the backend of the model.
    # For the TENSORRT-LLM backend,you currently must set 'exclude_input_in_output' to true
    # in the model config to not echo the input tokens
    backend: tensorrtllm

    # Set a custom endpoint that differs from the OpenAI defaults.
    custom:

    # The type to send requests to on the server.
    type: kserve

    # An option to enable the use of the streaming API.
    streaming: False

    # The list of server metrics URLs.
    # These are used for Telemetry metric reporting.
    server_metrics_urls: http://localhost:9400/metrics

    # URL of the endpoint to target for benchmarking.
    url: localhost:8001

    # A fully-qualified gRPC method name in '<package>.<service>/<method>' format.
    # The option is only supported by dynamic gRPC service kind and is
    # required to identify the RPC to use when sending requests to the server.
    grpc_method:
```

### Overriding Config Options
Once you have setup your config file to your liking, there could be times where you might want to re-profile with just a few options changed.</br>
Rather than editing your config you can include the `--override-config` option on the CLI along with the options you want to change. For example:
```bash
genai-perf config -f genai_perf_config.yaml --override-config --warmup-request-count 100 --concurrency 32
```
</br>

<!--
=====================
Analyze Subcommand
====================
-->
## Analyze
GenAI-Perf can be used to sweep through PA or GenAI-Perf stimulus allowing the user to profile multiple scenarios with a single command.
See [Analyze](docs/analyze.md) for details on how this subcommand can be utilized.

<!--
======================
VISUALIZATION
======================
-->

## Visualization

GenAI-Perf can also generate various plots that visualize the performance of the
current profile run. This is disabled by default but users can easily enable it
by passing the `--generate-plots` option when running the benchmark:

```bash
genai-perf profile \
  -m gpt2 \
  --backend tensorrtllm \
  --streaming \
  --concurrency 1 \
  --generate-plots
```

This will generate a [set of default plots](docs/example_plots.md) such as:
- Time to first token (TTFT) analysis
- Request latency analysis
- TTFT vs Input sequence lengths
- Inter token latencies vs Token positions
- Input sequence lengths vs Output sequence lengths

</br>

<!--
=====================
PROCESS EXPORT FILES SUBCOMMAND
====================
-->
## Process Export Files

GenAI-Perf can be used to process multiple profile export files from distributed runs and generate outputs with aggregated metrics.
See [Process Export Files](docs/process_export_files.md) for details on how this subcommand can be utilized.

</br>


<!--
======================
MODEL INPUTS
======================
-->

## Model Inputs

GenAI-Perf supports model input prompts from either synthetically generated
inputs, or from a dataset defined via a file.

When the dataset is synthetic, you can specify the following options:
* `--num-dataset-entries <int>`: The number of unique payloads to sample from.
  These will be reused until benchmarking is complete.
* `--synthetic-input-tokens-mean <int>`: The mean of number of tokens in the
  generated prompts when using synthetic data, >= 1.
* `--synthetic-input-tokens-stddev <int>`: The standard deviation of number of
  tokens in the generated prompts when using synthetic data, >= 0.
* `--random-seed <int>`: The seed used to generate random values, >= 0.
* `--request-count <int>`: The number of requests to benchmark
* `--warmup-request-count <int>`: The number of requests to send before
benchmarking

When the dataset is coming from a file, you can specify the following
options:
* `--input-file <path>`: The input file or directory containing the prompts or
  filepaths to images to use for benchmarking as JSON objects.

For any dataset, you can specify the following options:
* `--num-prefix-prompts <int>`: The number of synthetic prefix prompts to
  sample from. If this value is >0, synthetic prefix prompts will be prepended
  to user prompts.
* `--output-tokens-mean <int>`: The mean number of tokens in each output. Ensure
  the `--tokenizer` value is set correctly, >= 1.
* `--output-tokens-stddev <int>`: The standard deviation of the number of tokens
  in each output. This is only used when output-tokens-mean is provided, >= 1.
* `--output-tokens-mean-deterministic`: When using `--output-tokens-mean`, this
  flag can be set to improve precision by setting the minimum number of tokens
  equal to the requested number of tokens. This is currently supported with
  Triton. Note that there is still some variability in the
  requested number of output tokens, but GenAi-Perf attempts its best effort
  with your model to get the right number of output tokens.
* `--prefix-prompt-length <int>`: The number of tokens to include in each
  prefix prompt. This value is only used if --num-prefix-prompts is positive.

You can optionally set additional model inputs with the following option:
* `--extra-inputs <input_name>:<value>`: An additional input for use with the
  model with a singular value, such as `stream:true` or `max_tokens:5`. This
  flag can be repeated to supply multiple extra inputs.

For [Large Language Models](docs/tutorial.md), there is no batch size (i.e.
batch size is always `1`). Each request includes the inputs for one individual
inference. Other modes such as the [embeddings](docs/embeddings.md) and
[rankings](docs/rankings.md) endpoints support client-side batching, where
`--batch-size-text N` means that each request sent will include the inputs for
`N` separate inferences, allowing them to be processed together.

</br>

### Use moon_cake file as input payload

Genai-perf supports `--input-file payload:<file>` as the command option to use a payload file with a fixed schedule workload for profiling.

The payload file is in [moon_cake format](https://github.com/kvcache-ai/Mooncake) which contains a `timestamp` field and you can optionally add `input_length`, `output_length`, `text_input`, `session_id`, `hash_ids` and `priority`. For further examples using these fields, you can refer to our [multi-turn session benchmark tutorial](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/docs/multi_turn.md#approach-2-benchmark-with-a-custom-dataset).

Here is an example file:

```
{
    "timestamp": 0,
    "input_length": 6955,
    "output_length": 52,
    "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354]
}
{
    "timestamp": 10535,	# in milli-second
    "input_length": 6472,
    "output_length": 26,
    "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2366]
}
{
    "timestamp": 27482,
    "input_length": 6955,
    "output_length": 52,
    "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354]
}
```

`hash_ids` are a list of hash_id, each of which maps to a unique synthetic prompt sequence with `block_size` (current block size is 512) of tokens after tokenizer encoding, defined in [create_synthetic_prompt](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/genai_perf/inputs/retrievers/synthetic_prompt_generator.py#L37). Since the same hash_id maps to the same input block, it is effective to test features such as kv cache and speculative decoding.

### How to generate a payload file

#### 1. Synthetic data from sampling
[Nvidia Dynamo](https://github.com/ai-dynamo/dynamo) provides [a script](https://github.com/ai-dynamo/dynamo/blob/main/docs/guides/planner_benchmark/sin_synth.py) (with [README](https://github.com/ai-dynamo/dynamo/blob/main/docs/guides/planner_benchmark/benchmark_planner.md)) to generate synthetic moon_cake style payload:
```bash
python sin_synth.py \
    --time-duration 600 \
    --request-rate-min 5 \
    --request-rate-max 20 \
    --request-rate-period 150 \
    --isl1 3000 \
    --osl1 150 \
    --isl2 3000 \
    --osl2 150
```
This will generate a mooncake style payload file with
- duration = 600 seconds
- isl/osl = 3000/150
- request rate varies sinusoidally from 0.75 to 3 requests with a period of 150 seconds
For other models and GPU SKUs, adjust the request rate ranges accordingly to match the load.

Example genai-perf command to run the generated payload:
```bash
genai-perf profile \
    --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    -m deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --service-kind openai \
    --endpoint-type chat \
    --url http://localhost:8000 \
    --streaming \
    --input-file payload:sin_b512_t600_rr5.0-20.0-150.0_io3000150-3000150-0.2-0.8-10.jsonl
```

#### 2. Record real traffic and replay

We recommend users to build a traffic footprint collector over their inference service to generate the moon_cake format payload file based on real service traffic. Although popular inference servers do not support this feature, users can add this feature to their inference service.

Example code:
```python
# Hypothetical integration with the inference engine
import InferenceEngine  # Assume this exists

engine = InferenceEngine()
logger = Logger("mooncake_traffic.jsonl")

def process_request(prompt: str):
    result = engine.infer(prompt)
    logger.log_request({
        "input_length": len(engine.tokenize(prompt)),
        "output_length": len(engine.tokenize(result)),
        "text_input": prompt,
        "session_id": str(uuid.uuid4()),
        "hash_ids": engine.get_kvcache_hashes(prompt),
        "priority": 1
    })
    return result
```


<!--
======================
AUTHENTICATION
======================
-->

## Authentication

GenAI-Perf can benchmark secure endpoints such as OpenAI, which require API
key authentication. To do so, you must add your API key directly in the command.
Add the following flag to your command.

```bash
-H "Authorization: Bearer ${API_KEY}" -H "Accept: text/event-stream"
```

</br>

<!--
======================
METRICS
======================
-->

## Metrics

GenAI-Perf collects a diverse set of metrics that captures the performance of
the inference server.

| Metric | Description | Aggregations |
| - | - | - |
| <span id="time_to_first_token_metric">Time to First Token</span> | Time between when a request is sent and when its first response is received, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| <span id="time_to_second_token_metric">Time to Second Token</span> | Time between when the first streaming response is received and when the second streaming response is received, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| <span id="inter_token_latency_metric">Inter Token Latency</span> | Time between intermediate responses for a single request divided by the number of generated tokens of the latter response, one value per response per request in benchmark | Avg, min, max, p99, p90, p75 |
| <span id="output_token_throughput_per_user_metric">Output Token Throughput Per User</span> | Total number of output tokens (excluding the first token) divided by the total duration of the generation phase of each request | Avg, min, max, p99, p90, p75 |
| Request Latency | Time between when a request is sent and when its final response is received, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| Output Sequence Length | Total number of output tokens of a request, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| Input Sequence Length | Total number of input tokens of a request, one value per request in benchmark | Avg, min, max, p99, p90, p75 |
| <span id="output_token_throughput_metric">Output Token Throughput</span> | Total number of output tokens from benchmark divided by benchmark duration | None–one value per benchmark |
| <span id="request_throughput_metric">Request Throughput</span> | Number of final responses from benchmark divided by benchmark duration | None–one value per benchmark |

</br>

### GPU Telemetry

During benchmarking, GPU metrics such as GPU power usage, GPU utilization, energy consumption, and total GPU memory
are automatically collected. These metrics are collected from the `/metrics` endpoint
exposed by the [DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter), which must be running on the **same machine** as the inference server.

GenAI-Perf collects the following GPU metrics during benchmarking:

### Power Metrics
- `gpu_power_usage`
- `gpu_power_limit`
- `energy_consumption`

### Memory Metrics
- `total_gpu_memory`
- `gpu_memory_used`
- `gpu_memory_free`

### Temperature Metrics
- `gpu_temperature`
- `gpu_memory_temperature`

### Clock Metrics
- `gpu_clock_sm`
- `gpu_clock_memory`

### Utilization Metrics
- `gpu_utilization`
- `sm_utilization`
- `memory_copy_utilization`
- `video_encoder_utilization`
- `video_decoder_utilization`

> [!Note]
> Use the `--verbose` flag to print telemetry metrics on the console.

To collect GPU metrics, follow the [GPU Telmetry tutorial](docs/gpu_telemetry.md).
Use the provided [`custom-metrics.csv`](./docs/assets/custom_gpu_metrics.csv) to ensure all required metrics are included in the output.

</br>

<!--
======================
COMMAND LINE OPTIONS
======================
-->

## Command Line Options

##### `-h`
##### `--help`

Show the help message and exit.

### Config Options

##### `-f`
##### `--file`

The path to the config file - REQUIRED.

##### `--override-config`

An option that allows the user to override values specified in the config file.

### Template Options

##### `-f`
##### `--file`

The name of the template file to be created. Default is `genai_perf_config.yaml`.

### Endpoint Options:

##### `-m <list>`
##### `--model <list>`

The names of the models to benchmark.
A single model is recommended, unless you are
[profiling multiple LoRA adapters](docs/lora.md). (default: `None`)

##### `--model-selection-strategy {round_robin, random}`

When multiple models are specified, this is how a specific model
is assigned to a prompt. Round robin means that each model receives
a request in order. Random means that assignment is uniformly random
(default: `round_robin`)

##### `--backend {tensorrtllm,vllm}`

When benchmarking Triton, this is the backend of the model.
(default: tensorrtllm)

##### `--endpoint <str>`

Set a custom endpoint that differs from the OpenAI defaults. (default: `None`)

##### `--endpoint-type <str>`

The endpoint-type to send requests to on the server. (default: `kserve`)

##### `--server-metrics-urls <list>`

The list of server metrics URLs. These are used for Telemetry metric
reporting when benchmarking. Example usage: --server-metrics-urls
http://server1:9400/metrics http://server2:9400/metrics.
(default: `http://localhost:9400/metrics`)

##### `--streaming`

An option to enable the use of the streaming API. (default: `False`)

##### `-u <url>`
##### `--url <url>`

URL of the endpoint to target for benchmarking. (default: `None`)

##### `--grpc-method <str>`

A fully-qualified gRPC method name in '<package>.<service>/<method>' format.
The option is only supported with dynamic gRPC and is required to
identify the RPC to use when sending requests to the server. (default: `None`)

### Input Options

##### `--audio-length-mean <int>`

The mean length of audio data in seconds. (default: `0`)

##### `--audio-length-stddev <int>`

The standard deviation of the length of audio data in seconds.
(default: `0`)

##### `--audio-format <str>`

The format of the audio data. Currently we support wav and mp3 format.
(default: `wav`)

##### `--audio-depths <int>`

A list of audio bit depths to randomly select from in bits.
(default: `[16]`)

##### `--audio-sample-rates <int>`

A list of audio sample rates to randomly select from in kHz.
(default: `[16]`)

##### `--audio-num-channels <int>`

The number of audio channels to use for the audio data generation.
Currently only 1 (mono) and 2 (stereo) are supported.
(default: `1` (mono channel))

##### `-b <int>`
##### `--batch-size <int>`
##### `--batch-size-text <int>

The text batch size of the requests GenAI-Perf should send.
(default: `1`)
##### `--batch-size-text <int>

The text batch size of the requests GenAI-Perf should send.
(default: `1`)

##### `--batch-size-audio <int>`

The audio batch size of the requests GenAI-Perf should send.
This is currently only supported with the OpenAI `multimodal` endpoint type.
(default: `1`)

##### `--batch-size-text <int>`

The text batch size of the requests GenAI-Perf should send.
This is currently only supported with the
[embeddings](docs/embeddings.md), and
[rankings](docs/rankings.md) endpoint types.
(default: `1`)

##### `--batch-size-image <int>`

The image batch size of the requests GenAI-Perf should send.
This is currently only supported with the
image retrieval endpoint type.
(default: `1`)

##### `--extra-inputs <str>`

Provide additional inputs to include with every request. You can repeat this
flag for multiple inputs. Inputs should be in an input_name:value format.
Alternatively, a string representing a json formatted dict can be provided.
(default: `None`)

##### `--header <str>`
##### `--H <str>`
Add a custom header to the requests. Headers must be specified as
'Header:Value'. You can repeat this flag for multiple headers.
(default: `None`)

##### `--input-file <path>`

The input file or directory for profiling. Each line should be a JSON object
with a `text` or `image` field in JSONL format. Example:
`{"text": "Your prompt here"}`. To use synthetic files, prefix with
`synthetic:` followed by a comma-separated list of filenames without
extensions (Example: `synthetic:queries,passages`). To use a payload file with
a fixed schedule workload, prefix with `payload:` followed by the filename
(Example: `payload:input.jsonl`). (default: `None`)

##### `--num-dataset-entries <int>`

The number of unique payloads to sample from. These will be reused until
benchmarking is complete. (default: `100`)

##### `--num-prefix-prompts <int>`

The number of prefix prompts to select from. If this value is not zero, these
are prompts that are prepended to input prompts. This is useful for
benchmarking models that use a K-V cache. (default: `0`)

##### `--output-tokens-mean <int>`
##### `--osl`

The mean number of tokens in each output. Ensure the `--tokenizer` value is set
correctly. (default: `-1`)

##### `--output-tokens-mean-deterministic`

When using `--output-tokens-mean`, this flag can be set to improve precision by
setting the minimum number of tokens equal to the requested number of tokens.
This is currently supported with Triton. Note that there is
still some variability in the requested number of output tokens, but GenAi-Perf
attempts its best effort with your model to get the right number of output
tokens. (default: `False`)

##### `--output-tokens-stddev <int>`

The standard deviation of the number of tokens in each output. This is only used
when `--output-tokens-mean` is provided. (default: `0`)

##### `--random-seed <int>`

The seed used to generate random values. If not provided, a random seed will be
used.

##### `--request-count <int>`
##### `--num-requests <int>`

The number of requests to use for measurement.
(default: `10`)

##### `--synthetic-input-tokens-mean <int>`
##### `--isl`

The mean of number of tokens in the generated prompts when using synthetic
data. (default: `550`)

##### `--synthetic-input-tokens-stddev <int>`

The standard deviation of number of tokens in the generated prompts when
using synthetic data. (default: `0`)

##### `--prefix-prompt-length <int>`

The number of tokens in each prefix prompt. This value is only used if
--num-prefix-prompts is positive. Note that due to the prefix and user prompts
being concatenated, the number of tokens in the final prompt may be off by one.
(default: `100`)

##### `--image-width-mean <int>`

The mean width of images in pixels when generating synthetic image data.
(default: `0`)

##### `--image-width-stddev <int>`

The standard deviation of width of images in pixels when generating synthetic image data.
(default: `0`)

##### `--image-height-mean <int>`

The mean height of images in pixels when generating synthetic image data.
(default: `0`)

##### `--image-height-stddev <int>`

The standard deviation of height of images in pixels when generating synthetic image data.
(default: `0`)

##### `--image-format <str>`

The compression format of the images. If format is not selected,
format of generated image is selected at random.

##### `--warmup-request-count <int>`
##### `--num-warmup-requests <int>`

The number of warmup requests to send before benchmarking. (default: `0`)

### Profiling Options

##### `--concurrency <int>`

The concurrency value to benchmark. (default: `None`)

##### `--measurement-interval <int>`
##### `-p <int>`

The time interval used for each measurement in milliseconds.
Perf Analyzer will sample a time interval specified and take
measurement over the requests completed within that time interval.
When using the default stability percentage, GenAI-Perf will benchmark
for 3*(measurement_interval) milliseconds.

##### `--request-rate <float>`

Sets the request rate for the load generated by PA. (default: `None`)

##### `-s <float>`
##### `--stability-percentage <float>`

The allowed variation in latency measurements when determining if a result is
stable. The measurement is considered as stable if the ratio of max / min from
the recent 3 measurements is within (stability percentage) in terms of both
infer per second and latency. (default: `999`)

### Output Options

##### `--artifact-dir`

The directory to store all the (output) artifacts generated by GenAI-Perf and
Perf Analyzer. (default: `artifacts`)

##### `--generate-plots`

An option to enable the generation of plots. (default: False)

##### `--profile-export-file <path>`

The path where the perf_analyzer profile export will be generated. By default,
the profile export will be to `profile_export.json`. The genai-perf files will be
exported to `<profile_export_file>_genai_perf.json` and
`<profile_export_file>_genai_perf.csv`. For example, if the profile
export file is `profile_export.json`, the genai-perf file will be exported to
`profile_export_genai_perf.csv`. (default: `profile_export.json`)

### Session Options

##### `--num-sessions`

The number of sessions to simulate. This is used when generating synthetic
session data. (default: `0`)

##### `--session-concurrency <int>`

The number of concurrent sessions to benchmark. This must be specified
when benchmarking sessions. (default: `0`)

##### `--session-delay-ratio <float>`

A ratio to scale multi-turn delays when using a payload file. This allows adjusting
the timing between turns in a session without changing the payload file.
(default: `1.0`)

##### `--session-turn-delay-mean`

The mean delay (in milliseconds) between turns in a session.
(default: `0`)

##### `--session-turn-delay-stddev`

The standard deviation of the delay (in milliseconds) between turns in a session.
(default: `0`)

##### `--session-turns-mean`

The mean number of turns per session.
(default: `1`)

##### `--session-turns-stddev`

The standard deviation of the number of turns per session.
(default: `0`)

### Tokenizer Options

##### `--tokenizer <str>`

The HuggingFace tokenizer to use to interpret token metrics from prompts and
responses. The value can be the name of a tokenizer or the filepath of the
tokenizer. The default value is the model name.
(default: "<model_value>")

##### `--tokenizer-revision <str>`

The specific tokenizer model version to use. It can be a branch
name, tag name, or commit ID. (default: `main`)

##### `--tokenizer-trust-remote-code`

Allow custom tokenizer to be downloaded and executed. This carries security
risks and should only be used for repositories you trust. This is only
necessary for custom tokenizers stored in HuggingFace Hub.  (default: `False`)

### Other Options

##### `-v`
##### `--verbose`

An option to enable verbose mode. (default: `False`)

##### `--version`

An option to print the version and exit.

##### `-g <list>`
##### `--goodput <list>`

An option to provide constraints in order to compute goodput. Specify goodput
constraints as 'key:value' pairs, where the key is a valid metric name, and the
value is a number representing either milliseconds or a throughput value per
second. For example, 'request_latency:300' or
'output_token_throughput_per_user:600'. Multiple key:value pairs can be
provided, separated by spaces. (default: `None`)


</br>

<!--
======================
Known Issues
======================
-->

## Known Issues

* GenAI-Perf can be slow to finish if a high request-rate is provided
* Token counts may not be exact
