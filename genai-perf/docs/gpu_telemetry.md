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

# Collecting GPU Telemetry

This guide explains how to enable GPU metric collection during benchmarking with GenAI-Perf.
It also covers setting up the NVIDIA DCGM Exporter on the same machine as the inference server.

## Run the DCGM Exporter container

Create a custom GPU metrics file using the following command:

```
cat > custom_gpu_metrics.csv << 'EOF'
# Format
# If line starts with a '#' it is considered a comment
# DCGM FIELD, Prometheus metric type, help message

# Clocks
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)

# Temperature
DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temperature (in °C)
DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature (in °C)

# Power
DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W)
DCGM_FI_DEV_POWER_MGMT_LIMIT, gauge, Power management limit (in W)
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, counter, Total energy consumption since boot (in mJ)

# Memory usage
DCGM_FI_DEV_FB_FREE, gauge, Framebuffer memory free (in MiB)
DCGM_FI_DEV_FB_TOTAL, gauge, Total framebuffer memory (in MiB)
DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used (in MiB)

# Utilization
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)
DCGM_FI_DEV_MEM_COPY_UTIL, gauge, Memory copy utilization (in %)
DCGM_FI_DEV_ENC_UTIL, gauge, Encoder utilization (in %)
DCGM_FI_DEV_DEC_UTIL, gauge, Decoder utilization (in %)
DCGM_FI_PROF_SM_ACTIVE, gauge, Ratio of cycles at least one warp is active per SM
EOF
```
This will generate a `custom_gpu_metrics.csv` file that can be mounted into the DCGM Exporter container and
used directly for GPU telemetry collection with GenAI-Perf.

> [!Note]
> You can also view the complete metrics file at [custom_gpu_metrics.csv](./assets/custom_gpu_metrics.csv) for reference or direct use.

Start the DCGM Exporter using Docker on the same machine as your inference server:

```
docker run -d --gpus all --cap-add SYS_ADMIN \
  -p 9400:9400 \
  -v "$PWD/custom_gpu_metrics.csv:/etc/dcgm-exporter/custom.csv" \
  -e DCGM_EXPORTER_INTERVAL=33 \
  nvcr.io/nvidia/k8s/dcgm-exporter:4.2.0-4.1.0-ubuntu22.04 \
  -f /etc/dcgm-exporter/custom.csv
```

###  Configuration Details
#### Custom Collection Interval
By default, DCGM Exporter collects telemetry metrics every 30 seconds, which is too infrequent for detailed performance benchmarking.
GenAI-Perf expects metrics to be collected every 33 milliseconds for fine-grained profiling.
This is configured via the following environment variable:

```
-e DCGM_EXPORTER_INTERVAL=33
```

### Custom GPU Metrics
DCGM Exporter comes with a default set of metrics, but GenAI-Perf supports additional metrics that are not collected by default.

To collect all supported metrics,
you can either use the file you generated using the steps above, or refer to the provided [custom_gpu_metrics.csv](./assets/custom_gpu_metrics.csv) file.
Mount it into the container using:

```
-v "$PWD/custom_gpu_metrics.csv:/etc/dcgm-exporter/custom.csv"
```

> [!Note]
> You may comment out any metrics you do not want to collect in the CSV file.
> Lines starting with # are ignored by the exporter.

## Verifying DCGM Exporter is running
Once the container is running, confirm that metrics are being collected by running:

```
curl "localhost:9400/metrics"
```

You should see an output like this:

```
# HELP DCGM_FI_DEV_SM_CLOCK SM clock frequency (in MHz).
# TYPE DCGM_FI_DEV_SM_CLOCK gauge
# HELP DCGM_FI_DEV_MEM_CLOCK Memory clock frequency (in MHz).
# TYPE DCGM_FI_DEV_MEM_CLOCK gauge
# HELP DCGM_FI_DEV_MEMORY_TEMP Memory temperature (in C).
# TYPE DCGM_FI_DEV_MEMORY_TEMP gauge
...
DCGM_FI_DEV_SM_CLOCK{gpu="0", UUID="GPU-604ac76c-d9cf-fef3-62e9-d92044ab6e52"} 139
DCGM_FI_DEV_MEM_CLOCK{gpu="0", UUID="GPU-604ac76c-d9cf-fef3-62e9-d92044ab6e52"} 405
DCGM_FI_DEV_MEMORY_TEMP{gpu="0", UUID="GPU-604ac76c-d9cf-fef3-62e9-d92044ab6e52"} 9223372036854775794
...
```

> [!Note]
> For more details, see the [official DCGM Exporter documentation](https://github.com/NVIDIA/dcgm-exporter).

## Run GenAI-Perf

Once the DCGM Exporter is up and running, start benchmarking using GenAI-Perf.

Use the `--server-metrics-urls <list>` flag to specify one or more DCGM Exporter /metrics endpoints
from which GPU telemetry will be collected during benchmarking.

Example:

```
--server-metrics-urls http://localhost:9400/metrics http://remote-node:9400/metrics
```

By default, GenAI-Perf collects metrics from `http://localhost:9400/metrics`.

> [!Note]
> To enable printing GPU telemetry metrics on console, pass the `--verbose` or `-v` flag.

Example command:

```
genai-perf profile \
    -m gpt2 \
    --endpoint-type chat \
    --server-metrics-urls http://localhost:9400/metrics \
    --verbose
```

Example console output in `verbose` mode:

```
                 NVIDIA GenAI-Perf | Power Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                       GPU Power Usage (W)                        │
│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━┓ │
│ ┃ GPU Index ┃   avg ┃   min ┃    max ┃    p99 ┃    p90 ┃   p75 ┃ │
│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━┩ │
│ │         0 │ 63.55 │ 21.00 │ 111.39 │ 110.84 │ 105.91 │ 95.16 │ │
│ └───────────┴───────┴───────┴────────┴────────┴────────┴───────┘ │
│                 GPU Power Limit (W)                              │
│ ┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓             │
│ ┃ GPU Index ┃    avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃             │
│ ┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩             │
│ │         0 │ 300.00 │ N/A │ N/A │ N/A │ N/A │ N/A │             │
│ └───────────┴────────┴─────┴─────┴─────┴─────┴─────┘             │
│                    Energy Consumption (MJ)                       │
│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓    │
│ ┃ GPU Index ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃    │
│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩    │
│ │         0 │ 67.33 │ 67.33 │ 67.33 │ 67.33 │ 67.33 │ 67.33 │    │
│ └───────────┴───────┴───────┴───────┴───────┴───────┴───────┘    │
└──────────────────────────────────────────────────────────────────┘
               NVIDIA GenAI-Perf | Memory Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                     GPU Memory Used (GB)                      │
│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓ │
│ ┃ GPU Index ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃ │
│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩ │
│ │         0 │ 45.99 │ 45.99 │ 45.99 │ 45.99 │ 45.99 │ 45.99 │ │
│ └───────────┴───────┴───────┴───────┴───────┴───────┴───────┘ │
│                Total GPU Memory (GB)                          │
│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓           │
│ ┃ GPU Index ┃   avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃           │
│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩           │
│ │         0 │ 51.53 │ N/A │ N/A │ N/A │ N/A │ N/A │           │
│ └───────────┴───────┴─────┴─────┴─────┴─────┴─────┘           │
└───────────────────────────────────────────────────────────────┘
       NVIDIA GenAI-Perf | Utilization Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                GPU Utilization (%)                │
│ ┏━━━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓ │
│ ┃ GPU Index ┃ avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃ │
│ ┡━━━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩ │
│ │         0 │  28 │   0 │  50 │  50 │  50 │  49 │ │
│ └───────────┴─────┴─────┴─────┴─────┴─────┴─────┘ │
└───────────────────────────────────────────────────┘
```

Example output on a machine with multiple GPUs:

```
                NVIDIA GenAI-Perf | Power Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                      GPU Power Usage (W)                      │
│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓ │
│ ┃ GPU Index ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃ │
│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩ │
│ │         0 │ 79.60 │ 59.40 │ 98.85 │ 98.37 │ 94.03 │ 88.96 │ │
│ │         1 │ 42.09 │ 42.08 │ 42.10 │ 42.10 │ 42.10 │ 42.10 │ │
│ │         2 │ 43.99 │ 43.98 │ 44.00 │ 44.00 │ 44.00 │ 44.00 │ │
│ │         3 │ 42.56 │ 42.56 │ 42.56 │ 42.56 │ 42.56 │ 42.56 │ │
│ └───────────┴───────┴───────┴───────┴───────┴───────┴───────┘ │
│                 GPU Power Limit (W)                           │
│ ┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓          │
│ ┃ GPU Index ┃    avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃          │
│ ┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩          │
│ │         0 │ 300.00 │ N/A │ N/A │ N/A │ N/A │ N/A │          │
│ │         1 │ 300.00 │ N/A │ N/A │ N/A │ N/A │ N/A │          │
│ │         2 │ 300.00 │ N/A │ N/A │ N/A │ N/A │ N/A │          │
│ │         3 │ 300.00 │ N/A │ N/A │ N/A │ N/A │ N/A │          │
│ └───────────┴────────┴─────┴─────┴─────┴─────┴─────┘          │
│                 Energy Consumption (MJ)                       │
│ ┏━━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓       │
│ ┃ GPU Index ┃  avg ┃  min ┃  max ┃  p99 ┃  p90 ┃  p75 ┃       │
│ ┡━━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩       │
│ │         0 │ 0.28 │ 0.28 │ 0.28 │ 0.28 │ 0.28 │ 0.28 │       │
│ │         1 │ 0.23 │ 0.23 │ 0.23 │ 0.23 │ 0.23 │ 0.23 │       │
│ │         2 │ 0.25 │ 0.25 │ 0.25 │ 0.25 │ 0.25 │ 0.25 │       │
│ │         3 │ 0.24 │ 0.24 │ 0.24 │ 0.24 │ 0.24 │ 0.24 │       │
│ └───────────┴──────┴──────┴──────┴──────┴──────┴──────┘       │
└───────────────────────────────────────────────────────────────┘
               NVIDIA GenAI-Perf | Memory Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                     GPU Memory Used (GB)                      │
│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┓ │
│ ┃ GPU Index ┃   avg ┃   min ┃   max ┃   p99 ┃   p90 ┃   p75 ┃ │
│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━┩ │
│ │         0 │ 15.26 │ 15.26 │ 15.26 │ 15.26 │ 15.26 │ 15.26 │ │
│ │         1 │  0.00 │  0.00 │  0.00 │  0.00 │  0.00 │  0.00 │ │
│ │         2 │  0.00 │  0.00 │  0.00 │  0.00 │  0.00 │  0.00 │ │
│ │         3 │  0.00 │  0.00 │  0.00 │  0.00 │  0.00 │  0.00 │ │
│ └───────────┴───────┴───────┴───────┴───────┴───────┴───────┘ │
│                Total GPU Memory (GB)                          │
│ ┏━━━━━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓           │
│ ┃ GPU Index ┃   avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃           │
│ ┡━━━━━━━━━━━╇━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩           │
│ │         0 │ 17.18 │ N/A │ N/A │ N/A │ N/A │ N/A │           │
│ │         1 │ 17.18 │ N/A │ N/A │ N/A │ N/A │ N/A │           │
│ │         2 │ 17.18 │ N/A │ N/A │ N/A │ N/A │ N/A │           │
│ │         3 │ 17.18 │ N/A │ N/A │ N/A │ N/A │ N/A │           │
│ └───────────┴───────┴─────┴─────┴─────┴─────┴─────┘           │
└───────────────────────────────────────────────────────────────┘
       NVIDIA GenAI-Perf | Utilization Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│                GPU Utilization (%)                │
│ ┏━━━━━━━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┳━━━━━┓ │
│ ┃ GPU Index ┃ avg ┃ min ┃ max ┃ p99 ┃ p90 ┃ p75 ┃ │
│ ┡━━━━━━━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━╇━━━━━┩ │
│ │         0 │  34 │   0 │  51 │  51 │  51 │  51 │ │
│ │         1 │   0 │   0 │   0 │   0 │   0 │   0 │ │
│ │         2 │   0 │   0 │   0 │   0 │   0 │   0 │ │
│ │         3 │   0 │   0 │   0 │   0 │   0 │   0 │ │
│ └───────────┴─────┴─────┴─────┴─────┴─────┴─────┘ │
└───────────────────────────────────────────────────┘
```

> [!Note]
> GenAI-Perf prints a limited set of GPU metrics (as shown in the example output above) to the console
> when the --verbose (-v) flag is set. If GPU telemetry collection is configured correctly,
> all supported metrics enabled in the custom DCGM metrics file are always exported to CSV and JSON output files.
