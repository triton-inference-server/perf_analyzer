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

# GenAI-Perf Process Export Files Subcommand

The `process-export-files` subcommand is used to process multiple profile export files from distributed runs
and generate outputs with aggregated metrics.

## Process Export Files CLI
The `process-export-files` uses the following CLI options:

`--input-directory/-d` - The path to the input directory containing directories of profile export files from distributed runs. These directories must include perf analyzer profile export (e.g., profile_export.json) and GenAI-Perf profile export JSON files (e.g., profile_export_genai_perf.json).

Example input directory structure

```
input_dir/
    ├── run_1/
    │   ├── profile_export.json
    │   └── profile_export_genai_perf.json
    ├── run_2/
    │   ├── profile_export.json
    │   └── profile_export_genai_perf.json
    └── run_3/
        ├── profile_export.json
        └── profile_export_genai_perf.json
```
> [!Note]
> The file names can be anything as long as the files are of the correct type: one for perf analyzer profile data (\*.json) and
> one for GenAI-Perf profile data (\*_genai_perf.json).
> The names provided here (e.g., profile_export.json and profile_export_genai_perf.json) are just examples.

The `process-export-files` subcommand supports the following output options:

`--artifact-dir` - Specifies the directory where artifacts will be saved.

`--profile-export-file` - Custom name for the profile export files.

> [!Note]
> This subcommand does not support the `--generate-plots` option.

### CLI Examples
```bash
genai-perf process-export-files --input-directory /path/to/directory
```

Example output:

```
                               NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                            Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│             Time To First Token (ms) │  11.59 │   6.38 │  43.23 │  41.11 │  17.29 │  15.22 │
│            Time To Second Token (ms) │   4.13 │   3.84 │   7.22 │   6.67 │   4.05 │   4.00 │
│                 Request Latency (ms) │  71.89 │  64.63 │ 103.02 │ 102.05 │  80.83 │  74.94 │
│             Inter Token Latency (ms) │   4.03 │   3.88 │   4.39 │   4.37 │   4.26 │   4.02 │
│     Output Token Throughput Per User │ 248.14 │ 227.77 │ 257.95 │ 257.66 │ 256.09 │ 252.30 │
│                    (tokens/sec/user) │        │        │        │        │        │        │
│      Output Sequence Length (tokens) │  15.95 │  15.00 │  16.00 │  16.00 │  16.00 │  16.00 │
│       Input Sequence Length (tokens) │ 550.00 │ 550.00 │ 550.00 │ 550.00 │ 550.00 │ 550.00 │
│ Output Token Throughput (tokens/sec) │ 443.39 │    N/A │    N/A │    N/A │    N/A │    N/A │
│         Request Throughput (per sec) │  27.80 │    N/A │    N/A │    N/A │    N/A │    N/A │
│                Request Count (count) │  20.00 │    N/A │    N/A │    N/A │    N/A │    N/A │
└──────────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
```

This command processes profile export files from distributed runs located in subdirectories inside `/path/to/input/directory`.
It aggregates results across all runs and displays the aggregated metrics on the console.
The merged profile export file, along with GenAI-Perf JSON and CSV export files, are stored in the specified `artifacts` directory.

